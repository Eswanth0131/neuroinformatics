"""
NAFNet — Nonlinear Activation Free Network for Image Restoration.

Purpose-built for denoising/restoration tasks. Uses:
- SimpleGate: splits channels, element-wise multiply (replaces ReLU)
- Simplified Channel Attention (SCA): lightweight attention
- No nonlinear activations in the main path

Reference: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022)

Significantly fewer params than UNet while achieving better restoration quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return self.norm(x)


class SimpleGate(nn.Module):
    """Split channels in half, multiply. Replaces activation functions."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Pool → 1×1 conv → scale."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """
    Core NAFNet block:
    LayerNorm → 1×1 expand → 3×3 depthwise → SimpleGate → SCA → 1×1 project + skip
    """
    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        expanded = channels * expansion

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.dw_conv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate = SimpleGate()  # halves channels: expanded → expanded//2
        self.sca = SimplifiedChannelAttention(expanded // 2)
        self.conv2 = nn.Conv2d(expanded // 2, channels, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # FFN path
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, expanded, 1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(expanded // 2, channels, 1)
        self.ffn_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Spatial mixing
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.gate(x)
        x = self.sca(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x + shortcut

        # Channel mixing (FFN)
        shortcut = x
        x = self.norm2(x)
        x = self.ffn1(x)
        x = self.ffn_gate(x)
        x = self.ffn2(x)
        x = self.ffn_dropout(x)
        return x + shortcut


@register("nafnet")
class NAFNet(nn.Module):
    """
    NAFNet with UNet-like encoder-decoder structure.

    Default config (~7M params):
        width=48, enc_blocks=[1,1,2,4], dec_blocks=[1,1,1,1]

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W) with global residual
    """
    def __init__(
        self,
        in_channels=1,
        width=48,
        enc_blocks=(1, 1, 2, 4),
        dec_blocks=(1, 1, 1, 1),
        middle_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Input projection
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for num_blocks in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blocks)])
            )
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(middle_blocks)])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for num_blocks in dec_blocks:
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            ch //= 2
            # 1×1 conv to fuse skip connection (concat → project)
            self.fusions.append(nn.Conv2d(ch * 2, ch, 1))
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(num_blocks)])
            )

        # Output projection
        self.outro = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        # Residual from center channel
        if self.in_channels == 1:
            identity = x
        else:
            center = self.in_channels // 2
            identity = x[:, center:center+1, :, :]

        # Pad for divisibility
        _, _, h, w = x.shape
        factor = 2 ** len(self.encoders)
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        # Decoder
        for up, fusion, dec, skip in zip(self.ups, self.fusions, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = fusion(torch.cat([x, skip], dim=1))
            x = dec(x)

        x = self.outro(x)

        # Unpad
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return identity + x


@register("nafnet_25d")
class NAFNet25D(NAFNet):
    """2.5D NAFNet — 5-channel input."""
    def __init__(self, width=48, enc_blocks=(1, 1, 2, 4), dec_blocks=(1, 1, 1, 1), dropout=0.0):
        super().__init__(in_channels=5, width=width, enc_blocks=enc_blocks,
                         dec_blocks=dec_blocks, dropout=dropout)