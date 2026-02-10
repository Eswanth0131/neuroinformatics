"""
Competition Loss Functions — Full Stack

Components:
  - MS-SSIM: Multi-scale SSIM (replaces single-scale)
  - Perceptual: VGG-16 feature matching
  - Gradient: Sobel edge preservation
  - FFT: Frequency domain matching
  - L1: Pixel-level
  - Multi-scale supervision: apply combined loss at 1×, 1/2×, 1/4× resolution

Usage:
    loss_fn = FullLoss(device)
    loss = loss_fn(pred, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─── MS-SSIM ────────────────────────────────────────────────────────────────

class MSSSIM(nn.Module):
    """
    Multi-Scale SSIM.
    Computes SSIM at multiple resolutions, weighted average.
    More robust than single-scale for varying structure sizes.
    """
    def __init__(self, weights=(0.1, 0.2, 0.7), use_global=True):
        super().__init__()
        self.weights = weights  # coarse → fine
        self.use_global = use_global  # True = competition-style global SSIM

    def _ssim(self, pred, target):
        """Global SSIM matching Kaggle metric."""
        p = pred.reshape(pred.shape[0], -1).float()
        t = target.reshape(target.shape[0], -1).float()
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = p.mean(1, keepdim=True)
        mu_t = t.mean(1, keepdim=True)
        var_p = ((p - mu_p) ** 2).mean(1, keepdim=True)
        var_t = ((t - mu_t) ** 2).mean(1, keepdim=True)
        cov = ((p - mu_p) * (t - mu_t)).mean(1, keepdim=True)
        ssim = (2 * mu_p * mu_t + C1) * (2 * cov + C2) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
        return ssim.mean()

    def forward(self, pred, target):
        loss = 0
        p, t = pred.float(), target.float()
        for i, w in enumerate(self.weights):
            if i > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
            loss += w * (1 - self._ssim(p, t))
        return loss


# ─── Perceptual Loss ────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-16 feature matching loss.
    Compares intermediate CNN features rather than raw pixels.
    Captures texture and structural similarity that L1/SSIM miss.

    Uses layers: relu1_2, relu2_2, relu3_3 (shallow to mid — avoids
    high-level semantic features that don't apply to MRI).
    """
    def __init__(self, device, layer_weights=(1.0, 0.75, 0.5)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Extract features at specific layers
        self.blocks = nn.ModuleList()
        # relu1_2 = index 4, relu2_2 = index 9, relu3_3 = index 16
        breakpoints = [4, 9, 16]
        prev = 0
        for bp in breakpoints:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:bp + 1]))
            prev = bp + 1

        self.blocks = self.blocks.to(device)
        self.layer_weights = layer_weights

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        """Convert 1-channel MRI to 3-channel, normalize for VGG."""
        x = x.float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    @torch.no_grad()
    def _get_features(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, pred, target):
        pred_p = self._preprocess(pred)
        target_p = self._preprocess(target)
        pred_feats = self._get_features(pred_p)
        target_feats = self._get_features(target_p)

        loss = 0
        for pf, tf, w in zip(pred_feats, target_feats, self.layer_weights):
            loss += w * F.l1_loss(pf, tf)
        return loss


# ─── Gradient / Edge Loss ───────────────────────────────────────────────────

class GradientLoss(nn.Module):
    """Sobel edge preservation loss."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = kx.permute(0, 1, 3, 2)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _edges(self, x):
        ex = F.conv2d(x.float(), self.kx, padding=1)
        ey = F.conv2d(x.float(), self.ky, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


# ─── FFT Loss ───────────────────────────────────────────────────────────────

class FFTLoss(nn.Module):
    """Frequency domain magnitude matching."""
    def forward(self, pred, target):
        fp = torch.fft.rfft2(pred.float())
        ft = torch.fft.rfft2(target.float())
        return F.l1_loss(torch.abs(fp), torch.abs(ft))


# ─── Combined Multi-Scale Loss ──────────────────────────────────────────────

class FullLoss(nn.Module):
    """
    Full competition loss with multi-scale supervision.

    Combines:
      - MS-SSIM (replaces single-scale SSIM)
      - L1
      - Gradient (Sobel edges)
      - FFT (frequency domain)
      - Perceptual (VGG features)

    Applied at multiple resolutions (1×, 1/2×, 1/4×) for
    multi-scale supervision.

    Default weights calibrated for MRI restoration.
    """
    def __init__(self, device,
                 msssim_w=0.35, l1_w=0.25, grad_w=0.1,
                 fft_w=0.1, perceptual_w=0.2,
                 multiscale_weights=(0.7, 0.2, 0.1)):
        super().__init__()
        self.msssim = MSSSIM()
        self.grad = GradientLoss().to(device)
        self.fft = FFTLoss()
        self.perceptual = PerceptualLoss(device)

        self.msssim_w = msssim_w
        self.l1_w = l1_w
        self.grad_w = grad_w
        self.fft_w = fft_w
        self.perceptual_w = perceptual_w
        self.multiscale_weights = multiscale_weights  # 1×, 1/2×, 1/4×

    def _loss_at_scale(self, pred, target):
        """Compute combined loss at a single scale."""
        loss = self.msssim_w * self.msssim(pred, target)
        loss += self.l1_w * F.l1_loss(pred.float(), target.float())
        loss += self.grad_w * self.grad(pred, target)
        loss += self.fft_w * self.fft(pred, target)
        loss += self.perceptual_w * self.perceptual(pred, target)
        return loss

    def forward(self, pred, target):
        """Multi-scale supervised loss."""
        total = 0
        p, t = pred, target

        for i, w in enumerate(self.multiscale_weights):
            if i > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
            total += w * self._loss_at_scale(p, t)

        return total


# ─── Backward-compatible wrapper ────────────────────────────────────────────

class SimpleLoss(nn.Module):
    """
    Original loss without perceptual or multi-scale.
    For quick experiments or when VGG isn't available.
    """
    def __init__(self, ssim_w=0.4, l1_w=0.3, edge_w=0.15, fft_w=0.15):
        super().__init__()
        self.ssim = MSSSIM()
        self.grad = GradientLoss()
        self.fft = FFTLoss()
        self.ssim_w = ssim_w
        self.l1_w = l1_w
        self.edge_w = edge_w
        self.fft_w = fft_w

    def forward(self, pred, target):
        loss = self.ssim_w * self.ssim(pred, target)
        loss += self.l1_w * F.l1_loss(pred.float(), target.float())
        loss += self.edge_w * self.grad(pred, target)
        loss += self.fft_w * self.fft(pred, target)
        return loss
