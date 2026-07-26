"""
Paired data augmentation for grayscale image pairs.
"""

import random

import torch
from torch import Tensor


class PairedRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, src: Tensor, tgt: Tensor):
        if random.random() < self.p:
            return torch.flip(src, [-1]), torch.flip(tgt, [-1])
        return src, tgt


class PairedRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, src: Tensor, tgt: Tensor):
        if random.random() < self.p:
            return torch.flip(src, [-2]), torch.flip(tgt, [-2])
        return src, tgt


class PairedRandomRotation90:
    """Random rotation by 0, 90, 180, or 270 degrees."""

    def __call__(self, src: Tensor, tgt: Tensor):
        k = random.randint(0, 3)
        if k > 0:
            return torch.rot90(src, k, [-2, -1]), torch.rot90(tgt, k, [-2, -1])
        return src, tgt


class PairedCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, src: Tensor, tgt: Tensor):
        for t in self.transforms:
            src, tgt = t(src, tgt)
        return src, tgt


def build_transforms(augment: bool = True):
    if augment:
        return PairedCompose([
            PairedRandomHorizontalFlip(0.5),
            PairedRandomVerticalFlip(0.5),
            PairedRandomRotation90(),
        ])
    return PairedCompose([])
