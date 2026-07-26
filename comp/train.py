"""
Train U-Net baseline for DAPI -> CD68 image translation.

Usage:
    cd comp
    python train.py
    python train.py --config config.yaml --resume ../outputs/comp_baseline/checkpoints/epoch_040.pth
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ColonPairedDataset
from transforms import build_transforms
from models.unet import UNet
from models.losses import CombinedLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    set_seed(42)

    # --- Output dirs ---
    out = Path(cfg["training"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    # Save config for reproducibility
    import shutil
    shutil.copy(args.config, out / "config.yaml")

    # --- Dataset ---
    dc = cfg["data"]
    train_ds = ColonPairedDataset(dc["root"], dc["source"], dc["target"], dc["image_size"], "train", dc["train_ratio"])
    val_ds = ColonPairedDataset(dc["root"], dc["source"], dc["target"], dc["image_size"], "val", dc["train_ratio"])
    train_tf = build_transforms(augment=True)
    val_tf = build_transforms(augment=False)

    tc = cfg["training"]
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    print(f"[Train] Train: {len(train_ds)}, Val: {len(val_ds)}, Batches: {len(train_loader)}")

    # --- Model ---
    mc = cfg["model"]
    model = UNet(
        in_channels=mc["in_channels"],
        out_channels=mc["out_channels"],
        base_channels=mc["base_channels"],
        num_blocks=mc["num_blocks"],
        use_attention=mc["use_attention"],
        num_markers=1,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model params: {total:,}")

    # --- Loss ---
    loss_fn = CombinedLoss(cfg["loss"]).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["num_epochs"])

    start_epoch = 1
    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[Train] New params (random init): {len(missing)}")
        if unexpected:
            print(f"[Train] Unused params from ckpt: {len(unexpected)}")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("[Train] Optimizer state reset (model structure changed)")
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"[Train] Restarting at epoch {start_epoch}")

    # --- Fixed val batch for visualization ---
    val_dapi, val_cd68 = next(iter(val_loader))
    val_dapi, val_cd68 = val_dapi[:4].to(device), val_cd68[:4].to(device)

    # --- Training ---
    for epoch in range(start_epoch, tc["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for bi, (dapi, cd68) in enumerate(train_loader):
            dapi, cd68 = dapi.to(device), cd68.to(device)
            dapi_aug, cd68_aug = train_tf(dapi, cd68)

            pred = model(dapi_aug)
            loss, details = loss_fn(pred, cd68_aug)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

            if bi % tc["log_every_batches"] == 0:
                comps = " ".join(f"{k}={v:.4f}" for k, v in details.items())
                print(f"  E{epoch:3d} B{bi:4d} | {comps}")

        scheduler.step()
        avg_loss = epoch_loss / n
        print(f"[Train] Epoch {epoch:3d} | Avg Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- Validation + save ---
        if epoch % tc["save_every_epochs"] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = model(val_dapi)
                # Save comparison grid
                from torchvision.utils import save_image
                grid = torch.cat([val_dapi, val_cd68, preds], dim=0)  # 12 imgs
                grid = (grid + 1) / 2  # [-1,1] -> [0,1]
                save_image(grid, str(sample_dir / f"epoch_{epoch:03d}.png"), nrow=4)

                # Val loss
                val_loss = 0.0
                val_n = 0
                for vd, vt in val_loader:
                    vd, vt = vd.to(device), vt.to(device)
                    vd_t, vt_t = val_tf(vd, vt)
                    vl, _ = loss_fn(model(vd_t), vt_t)
                    val_loss += vl.item()
                    val_n += 1
                print(f"[Train] Epoch {epoch:3d} | Val Loss: {val_loss / val_n:.6f}")

            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, str(ckpt_dir / f"epoch_{epoch:03d}.pth"))
            print(f"[Train] Checkpoint saved: epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    main()
