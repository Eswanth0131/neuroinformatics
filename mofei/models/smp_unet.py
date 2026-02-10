"""
Pretrained Encoder UNet — leverages ImageNet features.

v2: EfficientNet-B4 encoder + SE attention on decoder.

Install: pip install segmentation-models-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from . import register


@register("smp_unet")
class SMPUNet(nn.Module):
    """ResNet34 encoder, basic decoder. Original version."""
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
        if self.in_channels == 1:
            identity = x
        else:
            center = self.in_channels // 2
            identity = x[:, center:center+1, :, :]
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
    """2.5D version — 5-channel input, ResNet34."""
    def __init__(self, encoder_name="resnet34", pretrained=True):
        super().__init__(encoder_name=encoder_name, in_channels=5, pretrained=pretrained)


@register("smp_unet_v2")
class SMPUNetV2(nn.Module):
    """
    Upgraded: EfficientNet-B4 encoder + scSE attention on decoder.
    More capacity, attention-gated skip connections.
    """
    def __init__(self, encoder_name="efficientnet-b4", in_channels=1, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )

    def forward(self, x):
        if self.in_channels == 1:
            identity = x
        else:
            center = self.in_channels // 2
            identity = x[:, center:center+1, :, :]
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.net(x)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return identity + out


@register("smp_unet_v2_25d")
class SMPUNetV225D(SMPUNetV2):
    """2.5D version — 5-channel input, EfficientNet-B4 + scSE attention."""
    def __init__(self, encoder_name="efficientnet-b4", pretrained=True):
        super().__init__(encoder_name=encoder_name, in_channels=5, pretrained=pretrained)
