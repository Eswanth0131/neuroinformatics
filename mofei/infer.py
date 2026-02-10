"""
Inference with test-time augmentation (TTA) → submission.csv

Supports both standard (1-channel) and 2.5D (N-channel) models.
TTA: predict on original + horizontally flipped, average both.

Usage:
    python infer.py --model unet --checkpoint runs/unet_final.pt --data_dir ./data
    python infer.py --model unet_25d --checkpoint runs/unet_25d_final.pt --data_dir ./data
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
    """Standard 2D: one slice at a time."""
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
    """2.5D: feed N adjacent slices, predict center."""
    model.eval()
    n_slices = volume.shape[2]
    half = n_adj // 2
    enhanced = np.zeros_like(volume)

    # Build all multi-slice inputs
    inputs = []
    for z in range(n_slices):
        slices = []
        for dz in range(-half, half + 1):
            idx = z + dz
            if idx < 0:
                idx = -idx
            elif idx >= n_slices:
                idx = 2 * (n_slices - 1) - idx
            idx = max(0, min(idx, n_slices - 1))
            slices.append(volume[:, :, idx])
        inputs.append(np.stack(slices, axis=0))  # (n_adj, H, W)

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch = np.stack(inputs[start:end], axis=0)  # (B, n_adj, H, W)
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not args.no_tta
    use_25d = args.model in IS_25D_MODELS

    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded {args.model} from {args.checkpoint}")
    print(f"TTA: {'enabled' if tta else 'disabled'}, Mode: {'2.5D' if use_25d else '2D'}")

    test_vols = preprocess_test(args.data_dir, args.cache_dir)

    predictions = {}
    for vol_data in test_vols:
        sample_id = vol_data["sample_id"]
        low_up = vol_data["low"]
        print(f"Enhancing {sample_id}...")
        if use_25d:
            enhanced = enhance_volume_25d(model, low_up, device, args.batch_size, tta=tta)
        else:
            enhanced = enhance_volume(model, low_up, device, args.batch_size, tta=tta)
        predictions[sample_id] = enhanced.astype(np.float64)

    df = create_submission_df(predictions)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
