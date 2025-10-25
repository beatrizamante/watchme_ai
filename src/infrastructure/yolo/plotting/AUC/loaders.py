"""
Module for loading YOLO prediction and ground truth label files and converting them to dictionaries
with absolute bounding box coordinates.
This module provides functions to:
- Load YOLO-format prediction files and convert normalized bounding boxes to absolute coordinates.
- Load YOLO-format ground truth label files and convert normalized bounding boxes to absolute coordinates.
"""
from collections import defaultdict
import glob
import os

from src.infrastructure.yolo.plotting.AUC.convert_boxes import yolo_xywh_to_xyxy

def load_predictions(pred_dir, img_sizes):
    """
    Loads YOLO-format predictions from text files in a specified directory and converts bounding boxes to absolute coordinates.
    Args:
        pred_dir (str): Path to the directory containing prediction text files. Each file should correspond to an image and contain predictions in YOLO format.
        img_sizes (dict): Dictionary mapping image IDs (str) to their sizes as (width, height) tuples.
    Returns:
        dict: A dictionary mapping image IDs to a list of predictions. Each prediction is a dictionary with keys:
            - 'box': List of absolute bounding box coordinates [x_min, y_min, x_max, y_max].
            - 'score': Confidence score (float).
            - 'class': Class label (int).
    """
    all_preds = defaultdict(list)
    for txt_file in glob.glob(os.path.join(pred_dir, "*.txt")):
        image_id = os.path.basename(txt_file).replace(".txt","")
        w,h = img_sizes[image_id]
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x_c, y_c, bw, bh, score = map(float, parts[:5])
                box = yolo_xywh_to_xyxy([x_c, y_c, bw, bh], w, h)
                all_preds[image_id].append({'box':box, 'score':score, 'class':cls})
    return all_preds


def load_ground_truth(gt_dir, img_sizes):
    """
    Loads ground truth bounding boxes and class labels from YOLO-format text files.
    Args:
        gt_dir (str): Directory containing ground truth .txt files, one per image.
        img_sizes (dict): Dictionary mapping image IDs to their (width, height) tuples.
    Returns:
        dict: A dictionary mapping image IDs to a list of ground truth objects,
              each represented as a dictionary with keys 'box' (list of [x1, y1, x2, y2])
              and 'class' (int class label).
    """
    all_gts = defaultdict(list)
    for txt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
        image_id = os.path.basename(txt_file).replace(".txt","")
        w,h = img_sizes[image_id]
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[:4])
                box = yolo_xywh_to_xyxy([x_c, y_c, bw, bh], w, h)
                all_gts[image_id].append({'box':box, 'class':cls})
    return all_gts
