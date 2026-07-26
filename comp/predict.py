"""
Inference script for DAPI -> CD68 prediction.

Usage:
    cd comp
    python predict.py --dapi ../dataset/dataset/colon/DAPI/00000.jpg \
        --checkpoint ../outputs/comp_baseline/checkpoints/epoch_200.pth \
        --output ../outputs/result.png

    python predict.py --dapi_dir ../dataset/dataset/colon/DAPI/ \
        --checkpoint ../outputs/comp_baseline/checkpoints/epoch_200.pth \
        --output_dir ../outputs/results/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models.unet import UNet


def load_image(path, size=256):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) / 127.5 - 1.0


def save_image(tensor, path):
    arr = (tensor.squeeze().clamp(-1, 1).cpu().numpy() + 1) / 2 * 255
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapi", type=str, default=None)
    parser.add_argument("--dapi_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="../outputs/result.png")
    parser.add_argument("--output_dir", type=str, default="../outputs/results/")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    mc = cfg["model"]
    model = UNet(
        in_channels=mc["in_channels"],
        out_channels=mc["out_channels"],
        base_channels=mc["base_channels"],
        num_blocks=mc["num_blocks"],
        use_attention=mc["use_attention"],
        num_markers=1,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"[Predict] Model loaded from {args.checkpoint}, epoch={ckpt.get('epoch', '?')}")

    size = cfg["data"]["image_size"]

    def predict_one(dapi_path, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        dapi = load_image(dapi_path, size).to(device)
        with torch.no_grad():
            pred = model(dapi)
        save_image(pred[0], out_path)
        print(f"  [OK] {Path(dapi_path).name} -> {out_path}")

    if args.dapi:
        predict_one(args.dapi, args.output)

    if args.dapi_dir:
        dapi_dir = Path(args.dapi_dir)
        files = sorted(list(dapi_dir.glob("*.jpg")) + list(dapi_dir.glob("*.jpg")) + list(dapi_dir.glob("*.png")))
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            predict_one(str(f), str(out_dir / f"cd68_{f.stem}.png"))


if __name__ == "__main__":
    main()
