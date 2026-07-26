import os
from huggingface_hub import snapshot_download

# 全局设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 完整下载整个模型仓库到 ./model/pix
snapshot_download(
    repo_id="StonyBrook-CVLab/PixCell-256",
    local_dir="./model/pix",
    force_download=True,
    resume_download=True,
    local_dir_use_symlinks=False
)
