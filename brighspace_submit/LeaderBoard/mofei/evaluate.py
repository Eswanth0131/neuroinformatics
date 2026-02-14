"""
Local evaluation — EXACT replica of Kaggle pipeline.

Runs the full: predict → encode (uint8) → decode → normalize → score
pipeline identically to what the competition server does.

Also supports cross-validation across multiple subject splits to
estimate true generalization performance.

Usage:
    # Single split (last 2 subjects):
    python evaluate.py --model unet --checkpoint runs/unet_best.pt --data_dir ./data

    # Cross-validation (rotate through all subjects):
    python evaluate.py --model unet --checkpoint runs/unet_best.pt --data_dir ./data --cross_val
"""

import argparse
import io
import base64
import numpy as np
import torch
from torch.amp import autocast

from dataset import preprocess_train, TARGET_SHAPE
from models import get_model


# ─── EXACT Kaggle encode/decode pipeline ─────────────────────────────────────

def slice_to_base64(slice_2d):
    """EXACT copy from extract_slices.py"""
    slice_min = float(slice_2d.min())
    slice_max = float(slice_2d.max())
    if slice_max - slice_min > 0:
        normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer,
                        slice=normalized,
                        shape=np.array(slice_2d.shape),
                        min_val=np.array([slice_min]),
                        max_val=np.array([slice_max]))
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_slice(b64_string):
    """EXACT copy from metric.py"""
    buffer = io.BytesIO(base64.b64decode(b64_string))
    data = np.load(buffer)
    normalized = data['slice']
    min_val_arr = data['min_val']
    max_val_arr = data['max_val']
    min_val = float(min_val_arr.item()) if min_val_arr.ndim > 0 else float(min_val_arr)
    max_val = float(max_val_arr.item()) if max_val_arr.ndim > 0 else float(max_val_arr)
    if max_val - min_val > 0:
        original = normalized.astype(np.float32) / 255 * (max_val - min_val) + min_val
    else:
        original = np.zeros_like(normalized, dtype=np.float32)
    return original


def encode_decode_roundtrip(slice_2d):
    """Run a slice through the full Kaggle encode→decode pipeline."""
    return base64_to_slice(slice_to_base64(slice_2d))


# ─── EXACT Kaggle metric functions ───────────────────────────────────────────

def normalize_01(x):
    """EXACT copy from metric.py compute_ssim/compute_psnr"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)


def kaggle_ssim(pred, gt):
    """EXACT copy of metric.py compute_ssim"""
    img1_norm = normalize_01(pred)
    img2_norm = normalize_01(gt)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1_norm.mean()
    mu2 = img2_norm.mean()
    sigma1_sq = ((img1_norm - mu1) ** 2).mean()
    sigma2_sq = ((img2_norm - mu2) ** 2).mean()
    sigma12 = ((img1_norm - mu1) * (img2_norm - mu2)).mean()
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(numerator / denominator)


def kaggle_psnr(pred, gt):
    """EXACT copy of metric.py compute_psnr"""
    img1_norm = normalize_01(pred)
    img2_norm = normalize_01(gt)
    mse = ((img1_norm - img2_norm) ** 2).mean()
    if mse == 0:
        return 50.0
    psnr = 10 * np.log10(1.0 / mse)
    return float(min(max(psnr, 0), 50))


def kaggle_score_slice(pred_slice, gt_slice, use_roundtrip=True):
    """
    Score a single slice using the EXACT Kaggle pipeline.

    If use_roundtrip=True, both pred and GT go through encode→decode
    (matching what Kaggle does — GT was encoded when creating solution.csv).
    """
    if use_roundtrip:
        pred_slice = encode_decode_roundtrip(pred_slice)
        gt_slice = encode_decode_roundtrip(gt_slice)

    s = kaggle_ssim(pred_slice, gt_slice)
    p = kaggle_psnr(pred_slice, gt_slice)
    return s, p


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


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_on_subjects(model, val_vols, device, batch_size=64, tta=True):
    """Evaluate model on a set of held-out subjects. Returns per-subject and overall scores."""
    all_ssim = []
    all_psnr = []
    subject_scores = []

    for vol in val_vols:
        sid = vol["sample_id"]
        enhanced = enhance_volume(model, vol["low"], device, batch_size, tta=tta)

        vol_ssim = []
        vol_psnr = []
        for z in range(TARGET_SHAPE[2]):
            s, p = kaggle_score_slice(
                enhanced[:, :, z].astype(np.float64),
                vol["high"][:, :, z].astype(np.float64),
                use_roundtrip=True,
            )
            vol_ssim.append(s)
            vol_psnr.append(p)

        mean_s = np.mean(vol_ssim)
        mean_p = np.mean(vol_psnr)
        score = 0.5 * mean_s + 0.5 * (mean_p / 50)
        subject_scores.append((sid, mean_s, mean_p, score))
        all_ssim.extend(vol_ssim)
        all_psnr.extend(vol_psnr)

    overall_ssim = np.mean(all_ssim)
    overall_psnr = np.mean(all_psnr)
    overall_score = 0.5 * overall_ssim + 0.5 * (overall_psnr / 50)

    return subject_scores, overall_ssim, overall_psnr, overall_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--val_subjects", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--cross_val", action="store_true",
                        help="Run leave-2-out cross-val across all subjects")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not getattr(args, "no_tta", False)

    # Load model
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded {args.model} from {args.checkpoint}")
    print(f"TTA: {'on' if tta else 'off'}")
    print(f"Pipeline: predict → uint8 encode → decode → normalize → score (exact Kaggle)")

    # Load all volumes
    volumes = preprocess_train(args.data_dir, args.cache_dir)

    if args.cross_val:
        # Leave-2-out cross-validation
        n = len(volumes)
        step = args.val_subjects
        all_scores = []
        print(f"\n--- Cross-validation (leave-{step}-out) ---")

        for start in range(0, n, step):
            end = min(start + step, n)
            val_vols = volumes[start:end]
            val_ids = [v["sample_id"] for v in val_vols]

            subj_scores, ov_ssim, ov_psnr, ov_score = evaluate_on_subjects(
                model, val_vols, device, args.batch_size, tta
            )
            for sid, ms, mp, sc in subj_scores:
                print(f"  {sid}: SSIM={ms:.4f}  PSNR={mp:.2f}dB  Score={sc:.4f}")
                all_scores.append(sc)

        print(f"\n{'='*50}")
        print(f"CROSS-VAL MEAN:   {np.mean(all_scores):.4f}")
        print(f"CROSS-VAL STD:    {np.std(all_scores):.4f}")
        print(f"CROSS-VAL MIN:    {np.min(all_scores):.4f}")
        print(f"CROSS-VAL MAX:    {np.max(all_scores):.4f}")
        print(f"{'='*50}")

    else:
        # Standard evaluation on last N subjects
        val_vols = volumes[-args.val_subjects:]
        print(f"\nEvaluating on {len(val_vols)} held-out subjects")

        subj_scores, ov_ssim, ov_psnr, ov_score = evaluate_on_subjects(
            model, val_vols, device, args.batch_size, tta
        )

        for sid, ms, mp, sc in subj_scores:
            print(f"  {sid}: SSIM={ms:.4f}  PSNR={mp:.2f}dB  Score={sc:.4f}")

        # Bicubic baseline
        print("\n--- Bicubic baseline ---")
        for vol in val_vols:
            sid = vol["sample_id"]
            bic_ssim, bic_psnr = [], []
            for z in range(TARGET_SHAPE[2]):
                s, p = kaggle_score_slice(
                    vol["low"][:, :, z].astype(np.float64),
                    vol["high"][:, :, z].astype(np.float64),
                    use_roundtrip=True,
                )
                bic_ssim.append(s)
                bic_psnr.append(p)
            ms, mp = np.mean(bic_ssim), np.mean(bic_psnr)
            sc = 0.5 * ms + 0.5 * (mp / 50)
            print(f"  {sid}: SSIM={ms:.4f}  PSNR={mp:.2f}dB  Score={sc:.4f}")

        print(f"\n{'='*50}")
        print(f"MODEL SCORE:  {ov_score:.4f}")
        print(f"  SSIM:       {ov_ssim:.4f}")
        print(f"  PSNR:       {ov_psnr:.2f} dB")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
