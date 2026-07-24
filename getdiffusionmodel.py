import os
import torch
from diffusers import StableDiffusionPipeline

# 模型保存路径（你指定的路径）
save_dir = r"C:\Users\16595\Desktop\aic\model\sd"
os.makedirs(save_dir, exist_ok=True)

# 拉取官方 Stable Diffusion v1.5（最常用、稳定）
model_id = "runwayml/stable-diffusion-v1-5"

print(f"开始拉取模型到：{save_dir}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=save_dir,  # 直接存到你指定的文件夹
    torch_dtype=torch.float32,
    safety_checker=None  # 可选：关闭安全检查器（如果不需要）
)

# 保存完整模型到本地，方便后续直接加载
pipe.save_pretrained(save_dir)

print(f"✅ Stable Diffusion 模型已拉取并保存到：{save_dir}")
