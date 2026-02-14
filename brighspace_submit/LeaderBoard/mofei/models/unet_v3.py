"""
UNet v3 — Regularized version of v1 UNet.

Same 31M param architecture but with:
- Dropout after each conv block (encoder + decoder)
- Dropout in bottleneck (heavier)
- Designed to close the train/val gap
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


@register("unet_v3")
class RegularizedUNet(nn.Module):
    """
    2D U-Net with global residual + dropout regularization.

    Dropout schedule:
      - Encoder: light (0.1) → heavier deeper
      - Bottleneck: heaviest (0.3)
      - Decoder: mirrors encoder
    """
    def __init__(self, base_ch=64, depth=4, dropout=0.15):
        super().__init__()
        self.depth = depth

        # Encoder — dropout increases with depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = 1
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            # Scale dropout: 0.5x at top, 1.5x at bottom
            d = dropout * (0.5 + i / max(depth - 1, 1))
            self.encoders.append(ConvBlock(ch, out_ch, dropout=d))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck — heaviest dropout
        self.bottleneck = ConvBlock(ch, ch * 2, dropout=dropout * 2)

        # Decoder — mirrors encoder dropout
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            d = dropout * (0.5 + i / max(depth - 1, 1))
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch, dropout=d))
            ch = out_ch

        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        identity = x

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
