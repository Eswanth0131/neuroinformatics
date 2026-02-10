"""
2.5D UNet — Uses adjacent slices as input channels for 3D context.

Instead of processing each slice independently (1-channel input),
feeds N neighboring slices (N-channel input) and predicts the center slice.

For brain MRI, adjacent slices are highly correlated. This gives the model
free 3D spatial context at zero architectural cost — same UNet, just
more input channels.

Default: 5 adjacent slices (2 above + center + 2 below)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


@register("unet_25d")
class UNet25D(nn.Module):
    """
    2.5D U-Net: multi-slice input, single-slice output.

    Input:  (B, n_adj, 179, 221) — n_adj adjacent slices
    Output: (B, 1, 179, 221)     — enhanced center slice

    Global residual: output = center_input + network(multi_slice_input)
    """
    def __init__(self, base_ch=64, depth=4, n_adj=5, dropout=0.0):
        super().__init__()
        self.depth = depth
        self.n_adj = n_adj

        # Encoder — first encoder takes n_adj channels instead of 1
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = n_adj  # <-- key difference from standard UNet
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            d = dropout * (0.5 + i / max(depth - 1, 1)) if dropout > 0 else 0
            self.encoders.append(ConvBlock(ch, out_ch, dropout=d))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2, dropout=dropout * 2 if dropout > 0 else 0)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            d = dropout * (0.5 + i / max(depth - 1, 1)) if dropout > 0 else 0
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch, dropout=d))
            ch = out_ch

        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # x: (B, n_adj, H, W)
        # Residual from center slice
        center_idx = self.n_adj // 2
        identity = x[:, center_idx:center_idx+1, :, :]  # (B, 1, H, W)

        # Pad for divisibility
        factor = 2 ** self.depth
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)
        x = x[:, :, :h, :w]

        return identity + x
