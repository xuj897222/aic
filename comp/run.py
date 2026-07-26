"""
Predict + Evaluate — single script.

Usage:
    # 单张预测+评分
    python run.py --checkpoint ../outputs/comp_baseline/checkpoints/epoch_006.pth --dapi ../dataset/dataset/colon/DAPI/00000.jpg --gt ../dataset/dataset/colon/CD68/00000.jpg

    # 全验证集评分
    python run.py --checkpoint ../outputs/comp_baseline/checkpoints/epoch_006.pth --eval

    # 全验证集评分+保存结果图
    python run.py --checkpoint ../outputs/comp_baseline/checkpoints/epoch_006.pth --eval --save
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.unet import UNet


# ============================================================
# Predict
# ============================================================

def load_image(path, size=256):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) / 127.5 - 1.0


def save_image(tensor, path):
    arr = (tensor.squeeze().clamp(-1, 1).cpu().numpy() + 1) / 2 * 255
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def load_model(checkpoint, config_path="config.yaml", device="cuda"):
    with open(config_path, encoding="utf-8") as f:
        mc = yaml.safe_load(f)["model"]
    model = UNet(in_channels=mc["in_channels"], out_channels=mc["out_channels"],
                 base_channels=mc["base_channels"], num_blocks=mc["num_blocks"],
                 use_attention=mc["use_attention"], num_markers=1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
    model.eval()
    print(f"[Model] {checkpoint}  epoch={ckpt.get('epoch', '?')}")
    return model


def predict_one(model, dapi_path, out_path, size=256):
    dapi = load_image(dapi_path, size).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(dapi)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(pred[0], out_path)
    return pred


# ============================================================
# Evaluate
# ============================================================

def load_gray(path, size=256):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(pred, gt, window_size=11):
    pred_t = torch.from_numpy(pred[None, None]).float()
    gt_t = torch.from_numpy(gt[None, None]).float()
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    w = (gauss[:, None] * gauss[None, :]).view(1, 1, window_size, window_size)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(pred_t, w, padding=window_size // 2)
    mu2 = F.conv2d(gt_t, w, padding=window_size // 2)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred_t * pred_t, w, padding=window_size // 2) - mu1_sq
    s2 = F.conv2d(gt_t * gt_t, w, padding=window_size // 2) - mu2_sq
    s12 = F.conv2d(pred_t * gt_t, w, padding=window_size // 2) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2) + 1e-8)
    return ssim_map.mean().item()


def norm_psnr(psnr, lo=10.0, hi=40.0):
    return np.clip((psnr - lo) / (hi - lo), 0.0, 1.0)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model .pth path")
    parser.add_argument("--dapi", type=str, default=None, help="Single DAPI image")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth CD68 for single evaluation")
    parser.add_argument("--eval", action="store_true", help="Evaluate on validation set")
    parser.add_argument("--save", action="store_true", help="Save prediction images when --eval")
    parser.add_argument("--num", type=int, default=0, help="Limit number of eval images (0=all)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.config, str(device))

    # --- 单张 ---
    if args.dapi and not args.eval:
        out = args.dapi.replace(".jpg", "_pred.png").replace(".png", "_pred.png")
        predict_one(model, args.dapi, out, args.size)
        print(f"[OK] -> {out}")
        if args.gt:
            pred_np = load_gray(out, args.size)
            gt_np = load_gray(args.gt, args.size)
            s = compute_ssim(pred_np, gt_np)
            p = compute_psnr(pred_np, gt_np)
            pn = norm_psnr(p)
            score = 0.7 * s + 0.3 * pn
            print(f"SSIM={s:.4f}  PSNR={p:.2f}dB  Score={score:.4f}")

    # --- 批量验证 ---
    elif args.eval:
        gt_dir = Path("../dataset/dataset/colon/CD68")
        dapi_dir = Path("../dataset/dataset/colon/DAPI")
        files = sorted([f.name for f in dapi_dir.iterdir() if f.suffix in (".jpg", ".png")])
        # Only validation split (last 10%)
        n = len(files)
        files = files[int(n * 0.9):]
        if args.num > 0:
            files = files[:args.num]

        out_dir = Path("../outputs/eval_run")
        out_dir.mkdir(parents=True, exist_ok=True)

        scores, ssims, psnrs = [], [], []
        for i, fn in enumerate(files):
            pred_path = out_dir / f"cd68_{Path(fn).stem}.png"
            if not args.save and pred_path.exists():
                pred_np = load_gray(str(pred_path), args.size)
            else:
                pred = predict_one(model, str(dapi_dir / fn), str(pred_path), args.size)
                pred_np = pred[0].squeeze().cpu().numpy()
                pred_np = (pred_np + 1) / 2  # [-1,1] -> [0,1]

            gt_np = load_gray(str(gt_dir / fn), args.size)
            s = compute_ssim(pred_np, gt_np)
            p = compute_psnr(pred_np, gt_np)
            pn = norm_psnr(p)
            sc = 0.7 * s + 0.3 * pn
            scores.append(sc); ssims.append(s); psnrs.append(p)
            bar = "=" * int(sc * 50)
            print(f"  [{i+1:3d}/{len(files)}] {fn}  SSIM={s:.4f} PSNR={p:.2f} Score={sc:.4f} {bar}")

        print(f"\n{'='*60}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Images:     {len(scores)}")
        print(f"SSIM:       {np.mean(ssims):.4f} +- {np.std(ssims):.4f}")
        print(f"PSNR:       {np.mean(psnrs):.2f} +- {np.std(psnrs):.2f} dB")
        print(f"SCORE:      {np.mean(scores):.4f} +- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
