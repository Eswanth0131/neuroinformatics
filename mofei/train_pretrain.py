"""
Two-stage training: pretrain on IXI synthetic pairs, fine-tune on competition data.

Stage 1 (pretrain): Learn general brain restoration from ~500 synthetic pairs
Stage 2 (finetune): Adapt to real 64mT characteristics from 18 competition pairs

Usage:
    # Full pipeline
    python train_pretrain.py --model smp_unet_25d --ixi_dir data/ixi_pairs \
        --data_dir ./data --pretrain_epochs 50 --finetune_epochs 100

    # Pretrain only (save checkpoint for later)
    python train_pretrain.py --model smp_unet_25d --ixi_dir data/ixi_pairs \
        --pretrain_epochs 50 --pretrain_only

    # Finetune from existing pretrained checkpoint
    python train_pretrain.py --model smp_unet_25d --data_dir ./data \
        --finetune_epochs 100 --pretrained_ckpt runs/smp_unet_25d_pretrained.pt
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel

from models import get_model
from dataset_ixi import get_ixi_dataloader

# Import competition dataloaders
from dataset import get_dataloaders
from dataset_25d import get_dataloaders_25d

IS_25D_MODELS = {"unet_25d", "smp_unet_25d", "nafnet_25d"}


# ─── Loss (same as train.py) ────────────────────────────────────────────────

class KaggleSSIMLoss(nn.Module):
    def forward(self, pred, target):
        pred_flat = pred.reshape(pred.shape[0], -1).float()
        tgt_flat = target.reshape(target.shape[0], -1).float()
        C1, C2 = 0.01**2, 0.03**2
        mu_p, mu_t = pred_flat.mean(1, keepdim=True), tgt_flat.mean(1, keepdim=True)
        var_p = ((pred_flat - mu_p)**2).mean(1, keepdim=True)
        var_t = ((tgt_flat - mu_t)**2).mean(1, keepdim=True)
        cov = ((pred_flat - mu_p) * (tgt_flat - mu_t)).mean(1, keepdim=True)
        ssim = (2*mu_p*mu_t + C1) * (2*cov + C2) / ((mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2))
        return (1 - ssim).mean()


def sobel_edges(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = kx.permute(0,1,3,2)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2 + 1e-6)


def fft_loss(pred, target):
    fp = torch.fft.rfft2(pred.float())
    ft = torch.fft.rfft2(target.float())
    return F.l1_loss(torch.abs(fp), torch.abs(ft))


def combined_loss(pred, target, ssim_loss_fn, ssim_w=0.4, l1_w=0.3, edge_w=0.15, fft_w=0.15):
    pred_f, tgt_f = pred.float(), target.float()
    loss = ssim_w * ssim_loss_fn(pred_f, tgt_f)
    loss += l1_w * F.l1_loss(pred_f, tgt_f)
    loss += edge_w * F.l1_loss(sobel_edges(pred_f), sobel_edges(tgt_f))
    loss += fft_w * fft_loss(pred_f, tgt_f)
    return loss


# ─── Training Loop ──────────────────────────────────────────────────────────

def train_stage(model, loader, optimizer, scheduler, scaler, ssim_loss_fn,
                epochs, stage_name, save_prefix, device, val_loader=None):
    """Generic training loop for both pretrain and finetune stages."""
    best_val = float('inf')
    print(f"\n{'='*60}")
    print(f"  {stage_name}: {epochs} epochs, {len(loader.dataset)} samples")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'LR':>10} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for batch_low, batch_high in loader:
            batch_low = batch_low.to(device, non_blocking=True)
            batch_high = batch_high.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(batch_low)
                loss = combined_loss(pred, batch_high, ssim_loss_fn)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / len(loader)
        elapsed = time.time() - t0

        # Validation
        val_str = "N/A"
        avg_val = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for vl, vh in val_loader:
                    vl, vh = vl.to(device), vh.to(device)
                    with autocast("cuda", dtype=torch.bfloat16):
                        vp = model(vl)
                        vl_loss = combined_loss(vp, vh, ssim_loss_fn)
                    val_loss += vl_loss.item()
            avg_val = val_loss / len(val_loader)
            val_str = f"{avg_val:.6f}"

        lr = optimizer.param_groups[0]['lr']
        print(f"  {epoch:3d} | {avg_train:10.6f} | {val_str:>10s} | {lr:10.2e} | {elapsed:5.1f}s")

        # Save logic
        save_metric = avg_val if avg_val is not None else avg_train
        if save_metric < best_val:
            best_val = save_metric
            torch.save(model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                       f"runs/{save_prefix}_best.pt")
            print(f"  ↳ Saved best model ({save_metric:.6f})")

        if epoch % 25 == 0 or epoch == epochs:
            torch.save(model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                       f"runs/{save_prefix}_ep{epoch}.pt")
            print(f"  ↳ Checkpoint (epoch {epoch})")

    # Save final
    torch.save(model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
               f"runs/{save_prefix}_final.pt")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="smp_unet_25d")
    parser.add_argument("--ixi_dir", type=str, default="data/ixi_pairs",
                        help="Directory with IXI synthetic .npz files")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Competition data directory")
    parser.add_argument("--cache_dir", type=str, default="./cache")

    # Stage control
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Skip pretrain, load this checkpoint for finetune")

    # Training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=2e-4)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val_subjects", type=int, default=2)
    parser.add_argument("--val_start", type=int, default=-1)
    parser.add_argument("--max_ixi_subjects", type=int, default=500)

    args = parser.parse_args()
    os.makedirs("runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_25d = args.model in IS_25D_MODELS
    mode = "25d" if use_25d else "2d"

    # Build model
    model = get_model(args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({n_params:,} params)")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    ssim_loss_fn = KaggleSSIMLoss()
    scaler = GradScaler("cuda")

    # ── Stage 1: Pretrain on IXI ──
    if args.pretrained_ckpt is None:
        if os.path.isdir(args.ixi_dir):
            print(f"\n=== STAGE 1: PRETRAIN on IXI ({args.ixi_dir}) ===")
            ixi_loader = get_ixi_dataloader(
                args.ixi_dir, batch_size=args.batch_size,
                num_workers=args.workers, augment=True,
                mode=mode, max_subjects=args.max_ixi_subjects,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.pretrain_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.pretrain_epochs, eta_min=1e-6,
            )

            train_stage(
                model, ixi_loader, optimizer, scheduler, scaler, ssim_loss_fn,
                epochs=args.pretrain_epochs,
                stage_name="PRETRAIN (IXI synthetic)",
                save_prefix=f"{args.model}_pretrained",
                device=device,
            )
            print(f"Pretrained model saved to runs/{args.model}_pretrained_final.pt")
        else:
            print(f"WARNING: IXI dir not found ({args.ixi_dir}), skipping pretrain")

    if args.pretrain_only:
        print("Pretrain only mode — done.")
        return

    # ── Load pretrained weights if provided ──
    if args.pretrained_ckpt:
        print(f"\nLoading pretrained checkpoint: {args.pretrained_ckpt}")
        state = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
        if isinstance(model, DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

    # ── Stage 2: Finetune on competition data ──
    print(f"\n=== STAGE 2: FINETUNE on competition data ({args.data_dir}) ===")

    if use_25d:
        train_loader, val_loader = get_dataloaders_25d(
            args.data_dir, cache_dir=args.cache_dir,
            val_subjects=args.val_subjects, val_start=args.val_start,
            batch_size=args.batch_size, num_workers=args.workers,
            augment=True,
        )
    else:
        train_loader, val_loader = get_dataloaders(
            args.data_dir, cache_dir=args.cache_dir,
            val_subjects=args.val_subjects, val_start=args.val_start,
            batch_size=args.batch_size, num_workers=args.workers,
            augment=True,
        )

    # Lower LR for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-6,
    )

    train_stage(
        model, train_loader, optimizer, scheduler, scaler, ssim_loss_fn,
        epochs=args.finetune_epochs,
        stage_name="FINETUNE (competition)",
        save_prefix=f"{args.model}_finetuned_split{args.val_start if args.val_start >= 0 else 'all'}",
        device=device,
        val_loader=val_loader,
    )

    print("\nDone! Ready for evaluation.")


if __name__ == "__main__":
    main()
