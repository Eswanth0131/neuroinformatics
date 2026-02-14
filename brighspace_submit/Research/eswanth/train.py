import os
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- import dataset loader ----
# dataset.py must be in same folder or on PYTHONPATH
try:
    from dataset import make_dataloaders
except Exception:
    make_dataloaders = None

# fallback if function is named differently
try:
    from dataset import get_dataloaders
except Exception:
    get_dataloaders = None


# ---- check if segmentation_models_pytorch is installed ----
def _require_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa
        return True
    except Exception:
        return False


class ResidualSMPUNet(nn.Module):
    # UNet with optional pretrained encoder + residual connection

    def __init__(self, encoder_name: str, in_channels: int = 1, pretrained: bool = True, decoder_attention_type: Optional[str] = None):
        super().__init__()

        # ensure SMP is installed
        if not _require_smp():
            raise ImportError(
                "segmentation-models-pytorch not installed.\n"
                "pip install -q segmentation-models-pytorch timm\n"
            )

        import segmentation_models_pytorch as smp

        self.in_channels = in_channels

        # build UNet backbone
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity branch for residual output
        if self.in_channels == 1:
            identity = x
        else:
            center = self.in_channels // 2
            identity = x[:, center:center + 1, :, :]

        _, _, h, w = x.shape

        # pad to multiple of 32 (UNet requirement)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out = self.net(x)

        # remove padding
        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        # residual output
        return identity + out


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    # choose model architecture
    name = name.lower().strip()

    if name == "smp_unet":
        return ResidualSMPUNet(
            encoder_name="resnet34",
            in_channels=1,
            pretrained=pretrained,
            decoder_attention_type=None
        )

    if name == "smp_unet_v2":
        return ResidualSMPUNet(
            encoder_name="efficientnet-b4",
            in_channels=1,
            pretrained=pretrained,
            decoder_attention_type="scse"
        )

    raise ValueError(f"Unknown model '{name}'.")


@dataclass
class Meter:
    # simple average tracker
    total: float = 0.0
    n: int = 0

    def update(self, value: float, count: int):
        self.total += value * count
        self.n += count

    @property
    def avg(self) -> float:
        return self.total / max(self.n, 1)


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, log_every: int = 50) -> float:
    # one full training pass
    model.train()
    meter = Meter()
    t0 = time.time()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # mixed precision if CUDA
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter.update(float(loss.item()), x.size(0))

        # periodic logging
        if log_every and step % log_every == 0:
            dt = time.time() - t0
            print(f"  step {step:05d} | train_loss={meter.avg:.6f} | {dt:.1f}s")

    return meter.avg


@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion) -> float:
    # validation loop
    model.eval()
    meter = Meter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        meter.update(float(loss.item()), x.size(0))

    return meter.avg


def save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float):
    # save model checkpoint
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    # argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="/content/drive/MyDrive/neuro/data")
    ap.add_argument("--model", type=str, default="smp_unet", choices=["smp_unet", "smp_unet_v2"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_subjects", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # build dataloaders
    if make_dataloaders is not None:
        train_loader, val_loader = make_dataloaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_subjects=args.val_subjects,
            cache_in_ram=True,
            augment=True,
            seed=args.seed,
        )
    elif get_dataloaders is not None:
        train_loader, val_loader = get_dataloaders(
            data_dir=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_subjects=args.val_subjects,
            augment=True,
            seed=args.seed,
        )
    else:
        raise ImportError("Could not import make_dataloaders or get_dataloaders.")

    if val_loader is None:
        raise ValueError("val_loader is None. Set --val_subjects >= 1")

    # build model
    model = build_model(args.model, pretrained=args.pretrained).to(device)

    # loss + optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")

    best_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
    last_path = os.path.join(args.save_dir, f"last_{args.model}.pt")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Saving best to: {best_path}")

    # training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, log_every=args.log_every)
        va = eval_one_epoch(model, val_loader, device, criterion)

        dt = time.time() - t0

        print(f"\nEpoch {epoch:03d}/{args.epochs} | train_mse={tr:.6f} | val_mse={va:.6f} | {dt/60:.1f} min\n")

        # save last checkpoint
        save_ckpt(last_path, model, optimizer, epoch, va)

        # save best checkpoint
        if va < best_val:
            best_val = va
            save_ckpt(best_path, model, optimizer, epoch, va)
            print(f"New best val_mse={best_val:.6f} -> {best_path}")

    print("Done.")


if __name__ == "__main__":
    main()