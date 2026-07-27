import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import VGG19_Weights


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset_sample" / "colon"
INPUT_DIR = DATA_ROOT / "DAPI"
TARGET_DIR = DATA_ROOT / "CD68"
OUTPUT_DIR = ROOT / "outputs" / "cd68"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.1
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairedImageDataset(Dataset):
    def __init__(self, input_dir: Path, target_dir: Path, image_size: int = IMAGE_SIZE, augment: bool = False) -> None:
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_size = image_size
        self.augment = augment
        self.image_names = [p.name for p in sorted(input_dir.glob("*.jpg"))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.image_names[idx]
        input_path = self.input_dir / name
        target_path = self.target_dir / name

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.augment:
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
                target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
                target_image = target_image.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.3:
                angle = random.choice([-10, -5, 5, 10])
                input_image = input_image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                target_image = target_image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            if random.random() < 0.4:
                color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)
                input_image = color_jitter(input_image)
            if random.random() < 0.2:
                noise_img = np.array(input_image, dtype=np.float32) / 255.0
                noise_img = noise_img + np.random.normal(0, 0.02, noise_img.shape)
                input_image = Image.fromarray((np.clip(noise_img, 0, 1) * 255).astype(np.uint8))
            if random.random() < 0.2:
                blur = transforms.GaussianBlur(kernel_size=3)
                input_image = blur(transforms.ToTensor()(input_image))
                input_image = transforms.ToPILImage()(input_image)

        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)
        return input_tensor, target_tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        def conv_block(in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        skip = F.interpolate(skip, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            AttentionBlock(32),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            AttentionBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            AttentionBlock(128),
        )
        self.pool = nn.MaxPool2d(2)

        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            AttentionBlock(256),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.fuse3 = FeatureFusionBlock(128)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            AttentionBlock(128),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fuse2 = FeatureFusionBlock(64)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            AttentionBlock(64),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse1 = FeatureFusionBlock(32)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            AttentionBlock(32),
        )
        self.out3 = nn.Conv2d(128, out_channels, 1)
        self.out2 = nn.Conv2d(64, out_channels, 1)
        self.out1 = nn.Conv2d(32, out_channels, 1)
        self.out_final = nn.Sequential(nn.Conv2d(32, out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bridge(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.fuse3(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.fuse2(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.fuse1(d1, e1)
        d1 = self.dec1(d1)

        output3 = F.interpolate(self.out3(d3), size=x.shape[2:], mode="bilinear", align_corners=False)
        output2 = F.interpolate(self.out2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
        output1 = self.out1(d1)
        output_final = self.out_final(d1)
        return output_final, output1, output2, output3


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu_x = F.avg_pool2d(img1, 3, 1, padding=1)
    mu_y = F.avg_pool2d(img2, 3, 1, padding=1)
    sigma_x2 = F.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu_x * mu_x
    sigma_y2 = F.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
    return (numerator / (denominator + 1e-8)).mean()


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0.0:
        return torch.tensor(float("inf"), device=img1.device)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
        except Exception:
            vgg = models.vgg19(weights=None).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.l1_loss(pred_features, target_features)


def compute_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    return 0.5 * (F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target))


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE) -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted([p.name for p in INPUT_DIR.glob("*.jpg")])
    if not all_files:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    val_count = max(1, int(len(all_files) * VAL_SPLIT))
    train_names = all_files[val_count:]
    val_names = all_files[:val_count]

    train_dataset = PairedImageDataset(INPUT_DIR, TARGET_DIR, image_size=IMAGE_SIZE, augment=True)
    val_dataset = PairedImageDataset(INPUT_DIR, TARGET_DIR, image_size=IMAGE_SIZE, augment=False)
    train_dataset.image_names = train_names
    val_dataset.image_names = val_names

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = EnhancedUNet().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)
    
    optimizer_g = optim.AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=lr * 0.1, betas=(0.5, 0.999))
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs)
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        progress = (epoch + 1) / epochs
        w_l1 = 1.0 - progress * 0.5
        w_mse = 0.1 * (1.0 - progress * 0.5)
        w_ssim = 0.1 * (1.0 - progress * 0.5)
        w_edge = 0.05
        w_percep = 0.05 + progress * 0.1
        w_adv = min(0.02, progress * 0.04)
        w_multi = 0.05 + progress * 0.05

        generator.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer_d.zero_grad()
            outputs, *_ = generator(inputs)
            
            real_pair = torch.cat([inputs, targets], dim=1)
            fake_pair = torch.cat([inputs, outputs.detach()], dim=1)
            
            real_pred = discriminator(real_pair)
            fake_pred = discriminator(fake_pair)
            
            loss_d_real = F.relu(1.0 - real_pred).mean()
            loss_d_fake = F.relu(1.0 + fake_pred).mean()
            loss_d = loss_d_real + loss_d_fake
            
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_d.step()
            running_loss_d += loss_d.item() * inputs.size(0)

            optimizer_g.zero_grad()
            outputs, aux1, aux2, aux3 = generator(inputs)
            
            fake_pair = torch.cat([inputs, outputs], dim=1)
            fake_pred = discriminator(fake_pair)
            
            l1 = l1_loss(outputs, targets)
            mse = mse_loss(outputs, targets)
            ssim = 1.0 - compute_ssim(outputs, targets)
            edge = compute_edge_loss(outputs, targets)
            perceptual = perceptual_loss(outputs, targets)
            multi_scale = l1_loss(aux1, targets) + l1_loss(aux2, targets) + l1_loss(aux3, targets)
            adv = -fake_pred.mean()
            
            loss_g = (
                w_l1 * l1
                + w_mse * mse
                + w_ssim * ssim
                + w_edge * edge
                + w_percep * perceptual
                + w_multi * multi_scale
                + w_adv * adv
            )
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer_g.step()
            running_loss_g += loss_g.item() * inputs.size(0)

        train_loss_g = running_loss_g / len(train_dataset)
        train_loss_d = running_loss_d / len(train_dataset)

        generator.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, *_ = generator(inputs)
                l1 = l1_loss(outputs, targets)
                mse = mse_loss(outputs, targets)
                ssim = 1.0 - compute_ssim(outputs, targets)
                edge = compute_edge_loss(outputs, targets)
                loss = l1 + 0.1 * mse + 0.1 * ssim + 0.05 * edge
                val_loss += loss.item() * inputs.size(0)
                val_psnr += compute_psnr(outputs, targets).item() * inputs.size(0)
                val_ssim += compute_ssim(outputs, targets).item() * inputs.size(0)

        val_loss = val_loss / len(val_dataset)
        val_psnr = val_psnr / len(val_dataset)
        val_ssim = val_ssim / len(val_dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} | G_loss={train_loss_g:.4f} | D_loss={train_loss_d:.4f} | "
            f"val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} | val_ssim={val_ssim:.4f}"
        )

        scheduler_g.step()
        scheduler_d.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "generator_state": {k: v.cpu().clone() for k, v in generator.state_dict().items()},
                "discriminator_state": {k: v.cpu().clone() for k, v in discriminator.state_dict().items()},
                "config": {"image_size": IMAGE_SIZE},
            }
            if (epoch + 1) % 5 == 0:
                torch.save(best_state, OUTPUT_DIR / f"checkpoint_epoch_{epoch + 1}.pth")

    checkpoint = best_state if best_state else {
        "generator_state": {k: v.cpu().clone() for k, v in generator.state_dict().items()},
        "config": {"image_size": IMAGE_SIZE},
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved best checkpoint to {CHECKPOINT_PATH}")


def infer(input_image_path: str, output_image_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedUNet().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    if "generator_state" in checkpoint:
        state = checkpoint["generator_state"]
    elif "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    
    model.load_state_dict(state)
    model.eval()

    image = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            output = outputs[0][0].cpu()
        else:
            output = outputs[0].cpu()

    output = output.permute(1, 2, 0).numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    Image.fromarray(output).save(output_image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a stronger 1-to-1 CD68 virtual staining model")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode to run: train or infer")
    parser.add_argument("--input", type=str, default=None, help="Path to an input DAPI image for inference")
    parser.add_argument("--output", type=str, default=None, help="Path to save the predicted HLA-DR image")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.mode == "infer":
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for inference")
        infer(args.input, args.output)
