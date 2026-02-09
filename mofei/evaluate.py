"""
Local evaluation using the EXACT Kaggle metric.

Runs model on held-out subject(s), computes per-slice global SSIM + PSNR
exactly as the competition does, reports the same score format.

Usage:
    python evaluate.py --model unet --checkpoint runs/unet_best.pt --data_dir ./data
"""

import argparse
import numpy as np
import torch
from torch.amp import autocast

from dataset import preprocess_train, TARGET_SHAPE
from models import get_model


# ─── Exact Kaggle metric functions ───────────────────────────────────────────

def normalize_01(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] — exact copy of competition metric."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)


def kaggle_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """EXACT competition SSIM: global statistics, not windowed."""
    pred_n = normalize_01(pred)
    gt_n = normalize_01(gt)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = pred_n.mean()
    mu2 = gt_n.mean()
    sigma1_sq = ((pred_n - mu1) ** 2).mean()
    sigma2_sq = ((gt_n - mu2) ** 2).mean()
    sigma12 = ((pred_n - mu1) * (gt_n - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def kaggle_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """EXACT competition PSNR: normalized, clamped to [0, 50]."""
    pred_n = normalize_01(pred)
    gt_n = normalize_01(gt)
    mse = ((pred_n - gt_n) ** 2).mean()
    if mse == 0:
        return 50.0
    psnr = 10 * np.log10(1.0 / mse)
    return float(min(max(psnr, 0), 50))


def kaggle_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Single-slice competition score."""
    return 0.5 * kaggle_ssim(pred, gt) + 0.5 * (kaggle_psnr(pred, gt) / 50)


# ─── Inference ────────────────────────────────────────────────────────────────

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--val_subjects", type=int, default=2,
                        help="Number of subjects to hold out for evaluation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not getattr(args, "no_tta", False)

    # Load model
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded {args.model} from {args.checkpoint}")

    # Load data — use last N subjects as eval
    volumes = preprocess_train(args.data_dir, args.cache_dir)
    val_vols = volumes[-args.val_subjects:]
    print(f"Evaluating on {len(val_vols)} held-out subjects")

    # Evaluate
    all_ssim = []
    all_psnr = []
    all_scores = []

    for vol in val_vols:
        sid = vol["sample_id"]
        low = vol["low"]
        gt = vol["high"]

        enhanced = enhance_volume(model, low, device, args.batch_size, tta=tta)

        vol_ssim = []
        vol_psnr = []
        for z in range(TARGET_SHAPE[2]):
            pred_slice = enhanced[:, :, z]
            gt_slice = gt[:, :, z]

            s = kaggle_ssim(pred_slice, gt_slice)
            p = kaggle_psnr(pred_slice, gt_slice)
            vol_ssim.append(s)
            vol_psnr.append(p)

        mean_s = np.mean(vol_ssim)
        mean_p = np.mean(vol_psnr)
        score = 0.5 * mean_s + 0.5 * (mean_p / 50)

        all_ssim.extend(vol_ssim)
        all_psnr.extend(vol_psnr)
        all_scores.append(score)

        print(f"  {sid}: SSIM={mean_s:.4f}  PSNR={mean_p:.2f}dB  Score={score:.4f}")

    # Also compute bicubic baseline
    print("\n--- Bicubic baseline ---")
    for vol in val_vols:
        sid = vol["sample_id"]
        low = vol["low"]
        gt = vol["high"]
        bic_ssim = []
        bic_psnr = []
        for z in range(TARGET_SHAPE[2]):
            s = kaggle_ssim(low[:, :, z], gt[:, :, z])
            p = kaggle_psnr(low[:, :, z], gt[:, :, z])
            bic_ssim.append(s)
            bic_psnr.append(p)
        mean_s = np.mean(bic_ssim)
        mean_p = np.mean(bic_psnr)
        score = 0.5 * mean_s + 0.5 * (mean_p / 50)
        print(f"  {sid}: SSIM={mean_s:.4f}  PSNR={mean_p:.2f}dB  Score={score:.4f}")

    # Overall
    overall_ssim = np.mean(all_ssim)
    overall_psnr = np.mean(all_psnr)
    overall_score = 0.5 * overall_ssim + 0.5 * (overall_psnr / 50)

    print(f"\n{'='*50}")
    print(f"MODEL SCORE:  {overall_score:.4f}")
    print(f"  SSIM:       {overall_ssim:.4f}")
    print(f"  PSNR:       {overall_psnr:.2f} dB")
    print(f"  (Kaggle formula: 0.5*{overall_ssim:.4f} + 0.5*{overall_psnr:.2f}/50)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
