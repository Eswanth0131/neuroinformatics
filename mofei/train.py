"""
Model-agnostic training loop.

Usage:
    python train.py --model unet_v2 --data_dir ./data --epochs 200
    python train.py --model unet_v2 --data_dir ./data --epochs 300 --lr 3e-4 --batch_size 64
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
from models import get_model


# ─── SSIM Loss (fp32-safe) ────────────────────────────────────────────────────

def gaussian_kernel(size=11, sigma=1.5, channels=1):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g /= g.sum()
    return g.view(1, 1, size, size).repeat(channels, 1, 1, 1)


class SSIMLoss(nn.Module):
    """SSIM loss that always computes in float32 for numerical stability."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.register_buffer("window", gaussian_kernel(window_size, sigma))

    def forward(self, pred, target):
        # Force fp32 for stability
        pred = pred.float()
        target = target.float()
        w = self.window

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = w.shape[-1] // 2

        mu_p = F.conv2d(pred, w, padding=pad, groups=1)
        mu_t = F.conv2d(target, w, padding=pad, groups=1)
        mu_pp, mu_tt, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_pp = F.conv2d(pred * pred, w, padding=pad, groups=1) - mu_pp
        sigma_tt = F.conv2d(target * target, w, padding=pad, groups=1) - mu_tt
        sigma_pt = F.conv2d(pred * target, w, padding=pad, groups=1) - mu_pt

        ssim = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim = self.ssim(pred, target)
        return (1 - self.ssim_weight) * l1 + self.ssim_weight * ssim


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16):
            pred = model(low)

        # Loss in fp32 (SSIM needs it, and it's cheap)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_subjects", type=int, default=2)
    parser.add_argument("--ssim_weight", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    # Data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        cache_dir=args.cache_dir,
        val_subjects=args.val_subjects,
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=True,
    )

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

    # Loss, optimizer, scheduler
    loss_fn = CombinedLoss(ssim_weight=args.ssim_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
    print(f"\n{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'LR':>10} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device) if val_loader else 0.0
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {lr:10.2e} | {dt:5.1f}s")

        # Save best
        if val_loader and val_loss < best_val:
            best_val = val_loss
            state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            path = os.path.join(args.run_dir, f"{save_name}_best.pt")
            torch.save(state, path)
            print(f"  ↳ Saved best model (val={best_val:.6f})")

        # Periodic checkpoint
        if epoch % 50 == 0:
            state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            path = os.path.join(args.run_dir, f"{save_name}_ep{epoch}.pt")
            torch.save(state, path)

    # Save final
    state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    path = os.path.join(args.run_dir, f"{save_name}_final.pt")
    torch.save(state, path)
    print(f"\nDone. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()