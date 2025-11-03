import json
import os

from tqdm import tqdm

coco_json = "src/dataset/yolo/labels/instances_val2017.json"
images_dir = "src/dataset/yolo/images/val"
labels_dir = "src/dataset/yolo/labels/val"
os.makedirs(labels_dir, exist_ok=True)

with open(coco_json, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

PERSON_CLASS_ID = 1

for ann in tqdm(coco["annotations"], desc="Converting"):
    if ann["category_id"] != PERSON_CLASS_ID:
        continue

    img_info = images[ann["image_id"]]
    img_w, img_h = img_info["width"], img_info["height"]

    x, y, w, h = ann["bbox"]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    txt_name = os.path.splitext(img_info["file_name"])[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_name)

    with open(txt_path, "a", encoding="utf-8") as f_out:
        f_out.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
