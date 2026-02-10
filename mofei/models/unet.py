import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@register("unet")
class ResidualUNet(nn.Module):
    """
    2D U-Net with global residual learning.
    Input: (B, 1, 179, 221) upsampled low-field slice
    Output: (B, 1, 179, 221) enhanced slice

    Learns residual: output = input + network(input)
    """

    def __init__(self, base_ch=64, depth=4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = 1
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))  # *2 for skip cat
            ch = out_ch

        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        identity = x

        # Pad to make dims divisible by 2^depth
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
            # Handle size mismatches from pooling
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)

        # Remove padding
        x = x[:, :, :h, :w]

        # Global residual
        return identity + x
