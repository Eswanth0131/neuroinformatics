"""
Evaluate model on specific held-out subjects via exact Kaggle pipeline.
Supports both 2D and 2.5D models.

Usage:
    python eval_fold.py --model unet --checkpoint runs/unet_split0_best.pt \
        --data_dir ./data --val_start 0 --val_subjects 2
    python eval_fold.py --model unet_25d --checkpoint runs/unet_25d_split0_best.pt \
        --data_dir ./data --val_start 0 --val_subjects 2
"""

import argparse
import io
import base64
import numpy as np
import torch
from torch.amp import autocast
from dataset import preprocess_train, TARGET_SHAPE
from models import get_model

IS_25D_MODELS = {"unet_25d", "smp_unet_25d", "nafnet_25d"}


def slice_to_base64(slice_2d):
    slice_min, slice_max = float(slice_2d.min()), float(slice_2d.max())
    if slice_max - slice_min > 0:
        normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, slice=normalized, shape=np.array(slice_2d.shape),
                        min_val=np.array([slice_min]), max_val=np.array([slice_max]))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def base64_to_slice(b64):
    buf = io.BytesIO(base64.b64decode(b64))
    d = np.load(buf)
    n, mn, mx = d['slice'], float(d['min_val'].item()), float(d['max_val'].item())
    return n.astype(np.float32) / 255 * (mx - mn) + mn if mx - mn > 0 else np.zeros_like(n, dtype=np.float32)


def roundtrip(s):
    return base64_to_slice(slice_to_base64(s))


def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(x)


def ssim(p, g):
    p, g = norm01(p), norm01(g)
    C1, C2 = 0.01**2, 0.03**2
    m1, m2 = p.mean(), g.mean()
    s1 = ((p - m1)**2).mean()
    s2 = ((g - m2)**2).mean()
    s12 = ((p - m1)*(g - m2)).mean()
    return float((2*m1*m2+C1)*(2*s12+C2)/((m1**2+m2**2+C1)*(s1+s2+C2)))


def psnr(p, g):
    p, g = norm01(p), norm01(g)
    mse = ((p - g)**2).mean()
    if mse == 0: return 50.0
    return float(min(max(10*np.log10(1.0/mse), 0), 50))


@torch.no_grad()
def enhance(model, vol, dev, tta=True):
    """Standard 2D inference."""
    model.eval()
    out = np.zeros_like(vol)
    for s in range(0, vol.shape[2], 64):
        e = min(s+64, vol.shape[2])
        b = torch.from_numpy(vol[:,:,s:e].transpose(2,0,1)[:,np.newaxis]).to(dev)
        with autocast("cuda", dtype=torch.bfloat16):
            p = model(b)
        p = p.float()
        if tta:
            with autocast("cuda", dtype=torch.bfloat16):
                pf = model(b.flip(-1))
            p = 0.5*p + 0.5*pf.float().flip(-1)
        out[:,:,s:e] = p.cpu().numpy()[:,0].transpose(1,2,0)
    return np.clip(out, 0, 1)


@torch.no_grad()
def enhance_25d(model, vol, dev, tta=True, n_adj=5):
    """2.5D inference with adjacent slices."""
    model.eval()
    n_slices = vol.shape[2]
    half = n_adj // 2
    out = np.zeros_like(vol)

    inputs = []
    for z in range(n_slices):
        slcs = []
        for dz in range(-half, half + 1):
            idx = z + dz
            if idx < 0: idx = -idx
            elif idx >= n_slices: idx = 2*(n_slices-1) - idx
            idx = max(0, min(idx, n_slices-1))
            slcs.append(vol[:,:,idx])
        inputs.append(np.stack(slcs, axis=0))

    for s in range(0, n_slices, 64):
        e = min(s+64, n_slices)
        b = torch.from_numpy(np.stack(inputs[s:e], axis=0)).float().to(dev)
        with autocast("cuda", dtype=torch.bfloat16):
            p = model(b)
        p = p.float()
        if tta:
            with autocast("cuda", dtype=torch.bfloat16):
                pf = model(b.flip(-1))
            p = 0.5*p + 0.5*pf.float().flip(-1)
        out[:,:,s:e] = p.cpu().numpy()[:,0].transpose(1,2,0)
    return np.clip(out, 0, 1)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True)
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--data_dir", default="./data")
    pa.add_argument("--cache_dir", default="./cache")
    pa.add_argument("--val_start", type=int, required=True)
    pa.add_argument("--val_subjects", type=int, default=2)
    args = pa.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_25d = args.model in IS_25D_MODELS

    m = get_model(args.model)
    m.load_state_dict(torch.load(args.checkpoint, map_location=dev, weights_only=True))
    m = m.to(dev)

    vols = preprocess_train(args.data_dir, args.cache_dir)
    val_vols = vols[args.val_start:args.val_start + args.val_subjects]

    scores = []
    for v in val_vols:
        sid = v["sample_id"]
        if use_25d:
            enh = enhance_25d(m, v["low"], dev)
        else:
            enh = enhance(m, v["low"], dev)
        ss, ps = [], []
        for z in range(TARGET_SHAPE[2]):
            p_rt = roundtrip(enh[:,:,z].astype(np.float64))
            g_rt = roundtrip(v["high"][:,:,z].astype(np.float64))
            ss.append(ssim(p_rt, g_rt))
            ps.append(psnr(p_rt, g_rt))
        ms, mp = np.mean(ss), np.mean(ps)
        sc = 0.5*ms + 0.5*(mp/50)
        scores.append(sc)
        print(f"  {sid}: SSIM={ms:.4f}  PSNR={mp:.2f}dB  Score={sc:.4f}")

    print(f"  MEAN: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
