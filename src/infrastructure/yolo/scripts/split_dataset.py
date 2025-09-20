import os
import random
import shutil
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    for split in ["train", "val", "test"]:
        (Path(output_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, files in splits.items():
        for img_file in files:
            label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
            
            shutil.copy(os.path.join(images_dir, img_file), Path(output_dir) / "images" / split / img_file)
            shutil.copy(os.path.join(labels_dir, label_file), Path(output_dir) / "labels" / split / label_file)

