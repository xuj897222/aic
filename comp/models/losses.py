"""
Loss functions for DAPI -> Marker image translation.

Includes: L1, L2, SSIM, Perceptual (VGG16), Edge, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# L1 + L2
# ============================================================

class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class L2Loss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


# ============================================================
# SSIM Loss
# ============================================================

class SSIMLoss(nn.Module):
    """1 - SSIM as loss. Window size 11, typical for SSIM."""

    def __init__(self, window_size: int = 11, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._gaussian_window(window_size, channels))

    def _gaussian_window(self, size, ch):
        sigma = 1.5
        gauss = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-gauss**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        window = window.expand(ch, 1, size, size).contiguous()
        return window

    def forward(self, pred, target):
        window = self.window.to(pred.device)
        mu1 = F.conv2d(pred, window, groups=self.channels, padding=self.window_size // 2)
        mu2 = F.conv2d(target, window, groups=self.channels, padding=self.window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, groups=self.channels, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, groups=self.channels, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, groups=self.channels, padding=self.window_size // 2) - mu12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        return 1 - ssim.mean()


# ============================================================
# Perceptual Loss (VGG16)
# ============================================================

class PerceptualLoss(nn.Module):
    """VGG16 perceptual loss on multiple layers."""

    def __init__(self, layers=(4, 9, 16, 23)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(vgg.features[prev:l + 1])
            prev = l + 1

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _to_rgb(self, x):
        """(B, 1, H, W) [-1,1] -> (B, 3, H, W) ImageNet normalized."""
        x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred = self._to_rgb(pred)
        target = self._to_rgb(target)
        loss = 0
        for sl in self.slices:
            pred = sl(pred)
            target = sl(target)
            loss += F.l1_loss(pred, target)
        return loss / len(self.slices)


# ============================================================
# Edge Loss (Sobel-based gradient consistency)
# ============================================================

class EdgeLoss(nn.Module):
    """L1 loss on image gradients (edges)."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

    def _grad(self, x):
        gx = F.conv2d(x, self.kernel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.kernel_y.to(x.device), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))


# ============================================================
# TV Loss (Total Variation — 抑制噪声/伪影)
# ============================================================

class TVLoss(nn.Module):
    """Total variation loss: penalize pixel-to-pixel variation."""

    def forward(self, x):
        return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
               (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple losses.

    Config:
        losses:
          l1: 1.0
          ssim: 0.5
          perceptual: 0.1
          edge: 0.2
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights
        self.losses = nn.ModuleDict()
        if "l1" in weights:
            self.losses["l1"] = L1Loss()
        if "l2" in weights:
            self.losses["l2"] = L2Loss()
        if "ssim" in weights:
            self.losses["ssim"] = SSIMLoss()
        if "perceptual" in weights:
            self.losses["perceptual"] = PerceptualLoss()
        if "edge" in weights:
            self.losses["edge"] = EdgeLoss()
        if "tv" in weights:
            self.losses["tv"] = TVLoss()

    def forward(self, pred, target):
        total = 0.0
        details = {}
        for name, loss_fn in self.losses.items():
            if name == "tv":
                val = loss_fn(pred)  # TV loss only needs prediction
            else:
                val = loss_fn(pred, target)
            w = self.weights[name]
            total += w * val
            details[name] = val.item()
        details["total"] = total.item()
        return total, details
