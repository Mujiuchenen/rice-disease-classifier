import os
import shutil
import random
from pathlib import Path

base_dir = Path("D:/rice_project/Rice/Rice/train")
labels_dir = Path("D:/rice_project/labels")
dest_dir = Path("D:/rice_project/detect_dataset")

for split in ['train', 'val']:
    (dest_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dest_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

all_images = []
for class_folder in base_dir.iterdir():
    if class_folder.is_dir():
        for img in class_folder.glob("*.jpg"):
            all_images.append(img)
        for img in class_folder.glob("*.png"):
            all_images.append(img)

print(f"找到 {len(all_images)} 张图片")

random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def copy_files(images, split):
    for img in images:
        dest_img = dest_dir / 'images' / split / img.name
        shutil.copy2(img, dest_img)
        label_file = labels_dir / f"{img.stem}.txt"
        if label_file.exists():
            dest_label = dest_dir / 'labels' / split / label_file.name
            shutil.copy2(label_file, dest_label)
        else:
            dest_label = dest_dir / 'labels' / split / f"{img.stem}.txt"
            dest_label.touch()

copy_files(train_images, 'train')
copy_files(val_images, 'val')

print(f"训练集: {len(train_images)} 张")
print(f"验证集: {len(val_images)} 张")