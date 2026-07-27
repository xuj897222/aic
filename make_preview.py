from pathlib import Path
import glob
from PIL import Image

root = Path(r"c:\vscode\projects\aic")
input_dir = root / "dataset_sample" / "colon" / "DAPI"
target_dir = root / "dataset_sample" / "colon" / "CD68"
out_dir = root / "outputs" / "generated"
preview_path = root / "outputs" / "preview_comparison.png"

files = sorted(glob.glob(str(input_dir / "*.jpg")))[:4]
width = 256
height = 256
canvas = Image.new("RGB", (width * 3, height * len(files)))

for row, image_path in enumerate(files):
    name = Path(image_path).name
    input_img = Image.open(image_path).convert("RGB").resize((width, height))
    generated_img = Image.open(out_dir / f"{Path(image_path).stem}.png").convert("RGB").resize((width, height))
    target_img = Image.open(target_dir / name).convert("RGB").resize((width, height))

    canvas.paste(input_img, (0, row * height))
    canvas.paste(generated_img, (width, row * height))
    canvas.paste(target_img, (width * 2, row * height))

canvas.save(preview_path)
print(f"Saved preview to {preview_path}")
