"""
Evaluate prediction quality: Score = 70% * SSIM + 30% * Normalize(PSNR)

Usage:
    python evaluate.py --pred ../outputs/comp_result.png --gt ../dataset/dataset/colon/CD68/00000.jpg
    python evaluate.py --pred_dir ../outputs/eval/ --gt_dir ../dataset/dataset/colon/CD68/ --pairs ../dataset/dataset/colon/DAPI/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
    """Pure PyTorch SSIM, avoids skimage dependency."""
    pred_t = torch.from_numpy(pred[None, None]).float()
    gt_t = torch.from_numpy(gt[None, None]).float()

    # Gaussian window
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :]).view(1, 1, window_size, window_size)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(pred_t, window, padding=window_size // 2)
    mu2 = F.conv2d(gt_t, window, padding=window_size // 2)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred_t * pred_t, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(gt_t * gt_t, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred_t * gt_t, window, padding=window_size // 2) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ssim_map.mean().item()


def normalize_psnr(psnr, psnr_min=10.0, psnr_max=40.0):
    return np.clip((psnr - psnr_min) / (psnr_max - psnr_min), 0.0, 1.0)


def compute_score(pred, gt, psnr_min=10.0, psnr_max=40.0):
    s = compute_ssim(pred, gt)
    p = compute_psnr(pred, gt)
    pn = normalize_psnr(p, psnr_min, psnr_max)
    score = 0.7 * s + 0.3 * pn
    return score, s, p, pn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--psnr_min", type=float, default=10.0)
    parser.add_argument("--psnr_max", type=float, default=40.0)
    args = parser.parse_args()

    def eval_one(pp, gp):
        return compute_score(load_gray(str(pp), args.size), load_gray(str(gp), args.size),
                             args.psnr_min, args.psnr_max)

    if args.pred and args.gt:
        score, s, p, pn = eval_one(args.pred, args.gt)
        print(f"SSIM:      {s:.4f}")
        print(f"PSNR:      {p:.2f} dB")
        print(f"PSNR_norm: {pn:.4f}")
        print(f"Score:     {score:.4f}")

    elif args.pred_dir and args.gt_dir:
        pred_dir = Path(args.pred_dir)
        gt_dir = Path(args.gt_dir)
        if args.pairs:
            files = sorted([f.name for f in Path(args.pairs).iterdir() if f.suffix in (".jpg", ".png")])
        else:
            files = sorted([f.name for f in pred_dir.iterdir() if f.suffix in (".jpg", ".png")])

        scores, ssims, psnrs = [], [], []
        for fn in files:
            # Try cd68_ prefix first, then direct filename
            pp = pred_dir / f"cd68_{Path(fn).stem}.png"
            if not pp.exists():
                pp = pred_dir / fn
            gp = gt_dir / fn
            if not pp.exists() or not gp.exists():
                print(f"  [SKIP] {fn}")
                continue
            score, s, p, pn = eval_one(pp, gp)
            scores.append(score); ssims.append(s); psnrs.append(p)
            print(f"  {fn}: SSIM={s:.4f}  PSNR={p:.2f}dB  Score={score:.4f}")

        if scores:
            print(f"\n{'='*50}")
            print(f"Total: {len(scores)} images")
            print(f"Avg SSIM:  {np.mean(ssims):.4f}  ± {np.std(ssims):.4f}")
            print(f"Avg PSNR:  {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
            print(f"Avg Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    else:
        print("Use --pred + --gt or --pred_dir + --gt_dir")
        print("SCORE = 0.7 * SSIM + 0.3 * (PSNR - PSNR_min) / (PSNR_max - PSNR_min)")


if __name__ == "__main__":
    main()
