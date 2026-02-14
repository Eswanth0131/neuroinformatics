"""
Ensemble inference — average predictions from multiple models.
Supports mixing 2D and 2.5D models in the same ensemble.

Usage:
    python ensemble_infer.py --data_dir ./data --output submission.csv \
        --models unet:runs/unet_best.pt unet_25d:runs/unet_25d_best.pt
"""

import argparse
import numpy as np
import torch
from torch.amp import autocast

from dataset import preprocess_test, TARGET_SHAPE
from extract_slices import create_submission_df
from models import get_model

IS_25D_MODELS = {"unet_25d", "smp_unet_25d", "nafnet_25d", "smp_unet_v2_25d"}


@torch.no_grad()
def enhance_volume(model, volume, device, batch_size=64, tta=True):
    model.eval()
    n_slices = volume.shape[2]
    enhanced = np.zeros_like(volume)
    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch = volume[:, :, start:end].transpose(2, 0, 1)[:, np.newaxis]
        batch_t = torch.from_numpy(batch).to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(batch_t)
        pred = out.float()
        if tta:
            with autocast("cuda", dtype=torch.bfloat16):
                out_flip = model(batch_t.flip(-1))
            pred = 0.5 * pred + 0.5 * out_flip.float().flip(-1)
        enhanced[:, :, start:end] = pred.cpu().numpy()[:, 0].transpose(1, 2, 0)
    return np.clip(enhanced, 0, 1)


@torch.no_grad()
def enhance_volume_25d(model, volume, device, batch_size=64, tta=True, n_adj=5):
    model.eval()
    n_slices = volume.shape[2]
    half = n_adj // 2
    enhanced = np.zeros_like(volume)
    inputs = []
    for z in range(n_slices):
        slcs = []
        for dz in range(-half, half + 1):
            idx = z + dz
            if idx < 0: idx = -idx
            elif idx >= n_slices: idx = 2*(n_slices-1) - idx
            idx = max(0, min(idx, n_slices-1))
            slcs.append(volume[:, :, idx])
        inputs.append(np.stack(slcs, axis=0))
    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch = np.stack(inputs[start:end], axis=0)
        batch_t = torch.from_numpy(batch).float().to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(batch_t)
        pred = out.float()
        if tta:
            with autocast("cuda", dtype=torch.bfloat16):
                out_flip = model(batch_t.flip(-1))
            pred = 0.5 * pred + 0.5 * out_flip.float().flip(-1)
        enhanced[:, :, start:end] = pred.cpu().numpy()[:, 0].transpose(1, 2, 0)
    return np.clip(enhanced, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="model_name:checkpoint_path pairs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not getattr(args, "no_tta", False)

    models = []
    for spec in args.models:
        model_name, ckpt_path = spec.split(":")
        model = get_model(model_name)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)
        models.append((model_name, model))
        print(f"Loaded {model_name} from {ckpt_path}")

    print(f"Ensemble of {len(models)} models, TTA={'on' if tta else 'off'}")

    test_vols = preprocess_test(args.data_dir, args.cache_dir)

    predictions = {}
    for vol_data in test_vols:
        sample_id = vol_data["sample_id"]
        low_up = vol_data["low"]
        print(f"Enhancing {sample_id}...")

        ensemble_pred = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2]), dtype=np.float64)
        for model_name, model in models:
            if model_name in IS_25D_MODELS:
                pred = enhance_volume_25d(model, low_up, device, args.batch_size, tta=tta)
            else:
                pred = enhance_volume(model, low_up, device, args.batch_size, tta=tta)
            ensemble_pred += pred

        ensemble_pred /= len(models)
        ensemble_pred = np.clip(ensemble_pred, 0, 1)
        predictions[sample_id] = ensemble_pred

    df = create_submission_df(predictions)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
