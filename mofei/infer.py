"""
Model-agnostic inference → submission.csv

Usage:
    python infer.py --model unet --checkpoint runs/unet_best.pt --data_dir ./data
"""

import argparse
import numpy as np
import torch
from torch.amp import autocast

from dataset import preprocess_test, TARGET_SHAPE
from extract_slices import create_submission_df
from models import get_model


@torch.no_grad()
def enhance_volume(model, volume: np.ndarray, device, batch_size=64) -> np.ndarray:
    """
    Run the model on all 200 axial slices of a (179,221,200) upsampled volume.
    Returns enhanced volume (179,221,200).
    """
    model.eval()
    n_slices = volume.shape[2]
    enhanced = np.zeros_like(volume)

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        # (B, 1, 179, 221)
        batch = volume[:, :, start:end].transpose(2, 0, 1)[:, np.newaxis]
        batch_t = torch.from_numpy(batch).to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(batch_t)

        out_np = out.float().cpu().numpy()[:, 0]  # (B, 179, 221)
        enhanced[:, :, start:end] = out_np.transpose(1, 2, 0)

    return np.clip(enhanced, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded {args.model} from {args.checkpoint}")

    # Load test data
    test_vols = preprocess_test(args.data_dir, args.cache_dir)

    # Generate predictions
    predictions = {}
    for vol_data in test_vols:
        sample_id = vol_data["sample_id"]
        low_up = vol_data["low"]
        print(f"Enhancing {sample_id}...")
        enhanced = enhance_volume(model, low_up, device, args.batch_size)
        predictions[sample_id] = enhanced.astype(np.float64)

    df = create_submission_df(predictions)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()