"""
Training loop optimized for Kaggle MRI SR competition.
Metric: 0.5 * SSIM + 0.5 * (PSNR / 50)

Key features:
  - Competition-aligned loss: SSIM + L1 + Edge (Sobel) + FFT
  - --val_subjects 0 for full-data training (saves periodically)
  - fp32 loss computation for stability with bf16 forward pass

Usage:
    # Full-data competition run:
    python train.py --model unet --data_dir ./data --epochs 200 --val_subjects 0

    # With validation (for development):
    python train.py --model swinir --data_dir ./data --epochs 200 --val_subjects 2
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel

from dataset import get_dataloaders
from dataset_25d import get_dataloaders_25d
from models import get_model

IS_25D_MODELS = {"unet_25d", "smp_unet_25d", "nafnet_25d"}  # models that need multi-slice input


# ─── Loss Components ──────────────────────────────────────────────────────────
# CRITICAL: Competition uses GLOBAL SSIM (whole-image statistics),
# NOT windowed SSIM (local 11×11 Gaussian). This was the key bug.

class KaggleSSIMLoss(nn.Module):
    """
    EXACT replica of the competition's SSIM as a differentiable loss.

    The competition normalizes pred and GT independently to [0,1],
    then computes SSIM with GLOBAL mean/variance (no local windows).
    """
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        # Per-sample independent normalization to [0,1]
        # (matches competition's normalize function)
        B = pred.shape[0]
        pred_flat = pred.view(B, -1)
        tgt_flat = target.view(B, -1)

        # Min-max normalize each sample independently
        p_min = pred_flat.min(dim=1, keepdim=True)[0]
        p_max = pred_flat.max(dim=1, keepdim=True)[0]
        t_min = tgt_flat.min(dim=1, keepdim=True)[0]
        t_max = tgt_flat.max(dim=1, keepdim=True)[0]

        p_range = (p_max - p_min).clamp(min=1e-8)
        t_range = (t_max - t_min).clamp(min=1e-8)

        pred_n = (pred_flat - p_min) / p_range
        tgt_n = (tgt_flat - t_min) / t_range

        # Global SSIM per sample
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_p = pred_n.mean(dim=1)
        mu_t = tgt_n.mean(dim=1)
        sigma_p_sq = ((pred_n - mu_p.unsqueeze(1)) ** 2).mean(dim=1)
        sigma_t_sq = ((tgt_n - mu_t.unsqueeze(1)) ** 2).mean(dim=1)
        sigma_pt = ((pred_n - mu_p.unsqueeze(1)) * (tgt_n - mu_t.unsqueeze(1))).mean(dim=1)

        num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2)
        ssim = num / den

        return 1.0 - ssim.mean()


class EdgeLoss(nn.Module):
    """Sobel edge loss — penalizes differences in gradient maps."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        tgt_gx = F.conv2d(target, self.sobel_x, padding=1)
        tgt_gy = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


class FFTLoss(nn.Module):
    """Frequency domain loss — penalizes errors in Fourier space."""
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(tgt_fft))


class CompetitionLoss(nn.Module):
    """
    Composite loss aligned with EXACT competition metric.

    Key change: uses global SSIM (matching Kaggle) instead of windowed SSIM.
    """
    def __init__(self, ssim_w=0.4, l1_w=0.3, edge_w=0.15, fft_w=0.15):
        super().__init__()
        self.ssim_loss = KaggleSSIMLoss()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.fft_loss = FFTLoss()
        self.ssim_w = ssim_w
        self.l1_w = l1_w
        self.edge_w = edge_w
        self.fft_w = fft_w

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        return (
            self.ssim_w * self.ssim_loss(pred, target)
            + self.l1_w * self.l1_loss(pred, target)
            + self.edge_w * self.edge_loss(pred, target)
            + self.fft_w * self.fft_loss(pred, target)
        )


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16):
            pred = model(low)

        loss = loss_fn(pred.float(), high.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * low.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            pred = model(low)
        loss = loss_fn(pred.float(), high.float())
        total_loss += loss.item() * low.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--run_dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_subjects", type=int, default=0,
                        help="0 = train on all data (competition mode)")
    parser.add_argument("--val_start", type=int, default=-1,
                        help="Index of first val subject. -1 = last N")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    # Loss weights
    parser.add_argument("--ssim_w", type=float, default=0.4)
    parser.add_argument("--l1_w", type=float, default=0.3)
    parser.add_argument("--edge_w", type=float, default=0.15)
    parser.add_argument("--fft_w", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    # Data
    print(f"Loading data from {args.data_dir}...")
    use_25d = args.model in IS_25D_MODELS
    if use_25d:
        train_loader, val_loader = get_dataloaders_25d(
            args.data_dir,
            cache_dir=args.cache_dir,
            val_subjects=args.val_subjects,
            val_start=args.val_start,
            batch_size=args.batch_size,
            num_workers=args.workers,
            augment=True,
            n_adj=5,
        )
    else:
        train_loader, val_loader = get_dataloaders(
            args.data_dir,
            cache_dir=args.cache_dir,
            val_subjects=args.val_subjects,
            val_start=args.val_start,
            batch_size=args.batch_size,
            num_workers=args.workers,
            augment=True,
        )

    if args.val_subjects == 0:
        print("** COMPETITION MODE: training on ALL 18 subjects, no validation **")

    # Model
    model = get_model(args.model)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({n_params:,} params)")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model = model.to(device)

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Loss
    loss_fn = CompetitionLoss(
        ssim_w=args.ssim_w, l1_w=args.l1_w,
        edge_w=args.edge_w, fft_w=args.fft_w,
    ).to(device)
    print(f"Loss: SSIM={args.ssim_w} L1={args.l1_w} Edge={args.edge_w} FFT={args.fft_w}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine schedule
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda")

    # Train
    best_val = float("inf")
    save_name = args.model
    if args.val_start >= 0:
        save_name = f"{args.model}_split{args.val_start}"
    has_val = val_loader is not None

    print(f"\n{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'LR':>10} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device) if has_val else 0.0
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        val_str = f"{val_loss:10.6f}" if has_val else "       N/A"
        print(f"{epoch:5d} | {train_loss:10.6f} | {val_str} | {lr:10.2e} | {dt:5.1f}s")

        state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()

        if has_val and val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(args.run_dir, f"{save_name}_best.pt"))
            print(f"  ↳ Saved best model (val={best_val:.6f})")

        # Periodic saves every 25 epochs
        if epoch % 25 == 0:
            torch.save(state, os.path.join(args.run_dir, f"{save_name}_ep{epoch}.pt"))
            print(f"  ↳ Checkpoint (epoch {epoch})")

    # Save final
    state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(state, os.path.join(args.run_dir, f"{save_name}_final.pt"))

    if has_val:
        print(f"\nDone. Best val loss: {best_val:.6f}")
    else:
        print(f"\nDone. Final train loss: {train_loss:.6f}")


if __name__ == "__main__":
    main()
