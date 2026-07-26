"""
Paired dataset: DAPI -> CD68 (grayscale, single-channel).
"""

import numpy as np
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


class ColonPairedDataset(Dataset):
    """
    DAPI -> CD68 paired grayscale dataset.

    root/
        DAPI/00000.jpg ... 01345.jpg
        CD68/00000.jpg ... 01345.jpg
    """

    def __init__(
        self,
        root: str = "dataset/dataset/colon",
        source: str = "DAPI",
        target: str = "CD68",
        image_size: int = 256,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.root = Path(root)
        self.source = source
        self.target = target
        self.image_size = image_size

        source_dir = self.root / source
        target_dir = self.root / target
        if not source_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {source_dir}")
        if not target_dir.exists():
            raise FileNotFoundError(f"Target dir not found: {target_dir}")

        self.files = sorted([
            f.name for f in source_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        # Verify pairing
        for fn in self.files:
            assert (target_dir / fn).exists(), f"Missing target: {target_dir / fn}"

        # Split
        n = len(self.files)
        split_idx = int(n * train_ratio)
        if split == "train":
            self.files = self.files[:split_idx]
        elif split == "val":
            self.files = self.files[split_idx:]
        # "all" keeps everything

        if split in ("train", "val"):
            print(f"[Dataset] {split}: {len(self.files)} pairs")

    def __len__(self):
        return len(self.files)

    def _load(self, path: Path) -> Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0) / 127.5 - 1.0  # [-1, 1]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        fn = self.files[idx]
        dapi = self._load(self.root / self.source / fn)
        cd68 = self._load(self.root / self.target / fn)
        return dapi, cd68
