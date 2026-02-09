"""
Pretrained Encoder UNet — leverages ImageNet features.

Uses segmentation_models_pytorch for a UNet with a pretrained encoder.
The encoder already knows edges, textures, contrast from ImageNet —
only needs to learn the low→high field mapping.

Install: pip install segmentation-models-pytorch

Supports both 1-channel (2D) and N-channel (2.5D) input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from . import register


@register("smp_unet")
class SMPUNet(nn.Module):
    """
    Pretrained UNet via segmentation_models_pytorch.

    Input:  (B, in_ch, H, W)
    Output: (B, 1, H, W)

    Global residual from center channel.
    """
    def __init__(self, encoder_name="resnet34", in_channels=1, pretrained=True):
        super().__init__()
        self.in_channels = in_channels

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        # x: (B, in_ch, H, W)
        if self.in_channels == 1:
            identity = x
        else:
            center = self.in_channels // 2
            identity = x[:, center:center+1, :, :]

        # Pad for encoder divisibility (typically needs /32)
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out = self.net(x)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return identity + out


@register("smp_unet_25d")
class SMPUNet25D(SMPUNet):
    """2.5D version — 5-channel input."""
    def __init__(self, encoder_name="resnet34", pretrained=True):
        super().__init__(encoder_name=encoder_name, in_channels=5, pretrained=pretrained)
