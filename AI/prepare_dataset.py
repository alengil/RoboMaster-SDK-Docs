import os
import shutil
import random
from pathlib import Path

DATASET_DIR = input("Enter dataset directory path: ").strip()
OUTPUT_DIR = input("Enter output directory path: ").strip()
TRAIN_SPLIT = float(input("Enter train split ratio (e.g., 0.8 for 80/20 split): ").strip())

os.makedirs(f"{OUTPUT_DIR}/train/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/train/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/valid/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/valid/labels", exist_ok=True)

images = list(Path(f"{DATASET_DIR}/images").glob("*.jpg")) + \
         list(Path(f"{DATASET_DIR}/images").glob("*.png"))

random.shuffle(images)

split_idx = int(len(images) * TRAIN_SPLIT)
train_images = images[:split_idx]
valid_images = images[split_idx:]

for img_path in train_images:
    label_path = Path(f"{DATASET_DIR}/labels/{img_path.stem}.txt")
    shutil.copy(img_path, f"{OUTPUT_DIR}/train/images/")
    if label_path.exists():
        shutil.copy(label_path, f"{OUTPUT_DIR}/train/labels/")

for img_path in valid_images:
    label_path = Path(f"{DATASET_DIR}/labels/{img_path.stem}.txt")
    shutil.copy(img_path, f"{OUTPUT_DIR}/valid/images/")
    if label_path.exists():
        shutil.copy(label_path, f"{OUTPUT_DIR}/valid/labels/")

classes_file = Path(DATASET_DIR) / "classes.txt"
if classes_file.exists():
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
else:
    classes = ["object"]

yaml_content = "path: {}\ntrain: train/images\nval: valid/images\n\nnames:\n".format(
    os.path.abspath(OUTPUT_DIR)
)
for i, cls in enumerate(classes):
    yaml_content += f"  {i}: {cls}\n"

with open(f"{OUTPUT_DIR}/data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✓ Dataset ready!")
print(f"  Train: {len(train_images)} images")
print(f"  Validation: {len(valid_images)} images")
print(f"  Path: {os.path.abspath(OUTPUT_DIR)}")
