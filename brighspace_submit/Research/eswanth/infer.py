import os
import re
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# utilities for loading test volumes + building submission CSV
from extract_slices import load_nifti, create_submission_df, NUM_SLICES


TARGET_SHAPE = (179, 221, 200)  # final expected (H, W, Z)
SAMPLE_RE = re.compile(r"(sample_\d+)")  # extract sample id from filename

# Preprocessing
def normalize_minmax(vol: np.ndarray) -> np.ndarray:
    # convert to float32 and scale to [0,1]
    vol = vol.astype(np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())

    # handle constant volume edge case
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def resize_3d_to_target(vol: np.ndarray, target_shape: Tuple[int, int, int] = TARGET_SHAPE) -> np.ndarray:
    # resize 3D volume using torch trilinear interpolation
    Ht, Wt, Zt = target_shape

    # convert (H,W,Z) -> (1,1,Z,H,W) for interpolate
    t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    t = F.interpolate(
        t,
        size=(Zt, Ht, Wt),
        mode="trilinear",
        align_corners=False
    )

    # convert back to (H,W,Z)
    out = t.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out.astype(np.float32)


def parse_sample_id(path: str) -> str:
    # extract sample id like "sample_001"
    base = os.path.basename(path)
    m = SAMPLE_RE.search(base)
    if not m:
        raise ValueError(f"Could not parse sample id from filename: {base}")
    return m.group(1)


def list_nii_files(folder: str) -> List[str]:
    # collect and sort all .nii/.nii.gz files
    files = []
    for fn in os.listdir(folder):
        if fn.endswith(".nii.gz") or fn.endswith(".nii"):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


# Model
class ResidualSMPUNet(nn.Module):
    # UNet backbone + residual output (identity + prediction)

    def __init__(self, encoder_name: str, in_channels: int = 1, pretrained: bool = False, decoder_attention_type=None):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.in_channels = in_channels

        # build UNet from SMP
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity branch for residual connection
        identity = x if self.in_channels == 1 else x[:, self.in_channels // 2:self.in_channels // 2 + 1]

        _, _, h, w = x.shape

        # pad to multiple of 32 (UNet requirement)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out = self.net(x)

        # remove padding after forward pass
        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        # residual output
        return identity + out


def build_model(model_name: str) -> nn.Module:
    # select architecture based on argument
    model_name = model_name.lower().strip()

    if model_name == "smp_unet":
        return ResidualSMPUNet(
            "resnet34",
            in_channels=1,
            pretrained=False,
            decoder_attention_type=None
        )

    if model_name == "smp_unet_v2":
        return ResidualSMPUNet(
            "efficientnet-b4",
            in_channels=1,
            pretrained=False,
            decoder_attention_type="scse"
        )

    raise ValueError("model_name must be one of: smp_unet, smp_unet_v2")

# Inference
@torch.no_grad()
def predict_volume(model: nn.Module, low_vol: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    # run slice-by-slice inference with batching
    H, W, Z = low_vol.shape

    # ensure resized correctly
    assert (H, W, Z) == TARGET_SHAPE

    out_vol = np.zeros((H, W, Z), dtype=np.float32)

    buf = []   # slice batch buffer
    zbuf = []  # track slice indices

    for z in range(Z):
        x = low_vol[:, :, z]  # extract 2D slice
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

        buf.append(x_t)
        zbuf.append(z)

        # run model when batch is full or last slice
        if len(buf) == batch_size or z == Z - 1:
            xb = torch.cat(buf, dim=0).to(device, non_blocking=True)

            # mixed precision if CUDA
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb)

            pred_np = pred.squeeze(1).float().cpu().numpy()

            # write predictions back to volume
            for i, zi in enumerate(zbuf):
                out_vol[:, :, zi] = pred_np[i]

            buf, zbuf = [], []

    return out_vol


def main():
    # argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="/content/drive/MyDrive/neuro/data")
    ap.add_argument("--ckpt", type=str, required=True)  # path to best checkpoint
    ap.add_argument("--model", type=str, required=True, choices=["smp_unet", "smp_unet_v2"])
    ap.add_argument("--out", type=str, default="/content/drive/MyDrive/neuro/submission.csv")
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # build model and load weights
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # support both full checkpoint dict or raw state_dict
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    # locate test data
    test_low_dir = os.path.join(args.data_root, "test", "low_field")
    if not os.path.isdir(test_low_dir):
        raise FileNotFoundError(f"Could not find test folder: {test_low_dir}")

    test_files = list_nii_files(test_low_dir)
    if len(test_files) == 0:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {test_low_dir}")

    predictions: Dict[str, np.ndarray] = {}

    # iterate over each test subject
    for fp in test_files:
        sid = parse_sample_id(fp)
        print(f"\nProcessing {sid}  ({os.path.basename(fp)})")

        low = load_nifti(fp)
        low = np.asarray(low, dtype=np.float32)

        # normalize + resize
        low = normalize_minmax(low)

        if tuple(low.shape) != TARGET_SHAPE:
            low = resize_3d_to_target(low, TARGET_SHAPE)

        # ensure correct slice count
        if low.shape[2] != NUM_SLICES:
            raise ValueError(f"{sid}: expected {NUM_SLICES} slices after resize, got {low.shape[2]}")

        # predict full 3D volume
        pred_vol = predict_volume(model, low, device=device, batch_size=args.batch_size)
        predictions[sid] = pred_vol

    # build submission file
    sub_df = create_submission_df(predictions)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub_df.to_csv(args.out, index=False)

    print(f"\nWrote submission: {args.out}")
    print(f"Rows: {len(sub_df)} (should be num_test_samples * {NUM_SLICES})")


if __name__ == "__main__":
    main()