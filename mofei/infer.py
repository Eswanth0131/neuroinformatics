"""
Inference with test-time augmentation (TTA) → submission.csv

TTA: predict on original + horizontally flipped, average both.
Typically adds +0.01-0.02 SSIM for free.

Usage:
    python infer.py --model unet --checkpoint runs/unet_final.pt --data_dir ./data
    python infer.py --model unet --checkpoint runs/unet_final.pt --data_dir ./data --no-tta
"""

import argparse
import numpy as np
import torch
from torch.amp import autocast

from dataset import preprocess_test, TARGET_SHAPE
from extract_slices import create_submission_df
from models import get_model


@torch.no_grad()
def enhance_volume(model, volume: np.ndarray, device, batch_size=64, tta=True) -> np.ndarray:
    """
    Run model on all axial slices with optional TTA.
    volume: (179, 221, 200)
    Returns: enhanced volume (179, 221, 200)
    """
    model.eval()
    n_slices = volume.shape[2]
    enhanced = np.zeros_like(volume)

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        # (B, 1, 179, 221)
        batch = volume[:, :, start:end].transpose(2, 0, 1)[:, np.newaxis]
        batch_t = torch.from_numpy(batch).to(device)

        # Forward pass
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(batch_t)
        pred = out.float()

        if tta:
            # Horizontal flip
            with autocast("cuda", dtype=torch.bfloat16):
                out_flip = model(batch_t.flip(-1))
            pred = 0.5 * pred + 0.5 * out_flip.float().flip(-1)

        out_np = pred.cpu().numpy()[:, 0]  # (B, 179, 221)
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
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not args.no_tta

    # Load model
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded {args.model} from {args.checkpoint}")
    print(f"TTA: {'enabled' if tta else 'disabled'}")

    # Load test data
    test_vols = preprocess_test(args.data_dir, args.cache_dir)

    # Generate predictions
    predictions = {}
    for vol_data in test_vols:
        sample_id = vol_data["sample_id"]
        low_up = vol_data["low"]
        print(f"Enhancing {sample_id}...")
        enhanced = enhance_volume(model, low_up, device, args.batch_size, tta=tta)
        predictions[sample_id] = enhanced.astype(np.float64)

    df = create_submission_df(predictions)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
