"""
This module provides functionality to match predicted bounding boxes to ground truth boxes and build lists of scores and labels for each class, facilitating the calculation of metrics such as AUC for object detection tasks.
"""
from collections import defaultdict

from src.infrastructure.yolo.plotting.AUC.calculate_iou import box_iou


def build_scores_labels(all_preds, all_gts, iou_threshold=0.5, class_id=None):
    """
    Matches predicted bounding boxes to ground truth boxes and builds lists of scores and labels for each class.
    Args:
        all_preds (dict): Dictionary mapping image IDs to lists of predicted boxes. Each prediction is a dict with keys 'box', 'score', and 'class'.
        all_gts (dict): Dictionary mapping image IDs to lists of ground truth boxes. Each ground truth is a dict with keys 'box' and 'class'.
        iou_threshold (float, optional): Minimum IoU required to consider a prediction as a true positive. Defaults to 0.5.
        class_id (int, optional): If specified, only predictions and ground truths of this class are considered. Defaults to None.
    Returns:
        tuple:
            - per_class_scores (defaultdict(list)): Dictionary mapping class IDs to lists of prediction scores.
            - per_class_labels (defaultdict(list)): Dictionary mapping class IDs to lists of labels (1 for true positive, 0 for false positive).
    """
    per_class_scores = defaultdict(list)
    per_class_labels = defaultdict(list)

    for img_id, preds in all_preds.items():
        gts = all_gts.get(img_id, [])
        if class_id is not None:
            preds = [p for p in preds if p['class']==class_id]
            gts = [g for g in gts if g['class']==class_id]

        gts_by_class = defaultdict(list)
        for g in gts:
            gts_by_class[g['class']].append(g['box'])
        matched_flags = {c:[False]*len(gts_by_class[c]) for c in gts_by_class}

        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        for p in preds_sorted:
            cls = p['class']
            best_iou = 0
            best_idx = -1
            for i, gt_box in enumerate(gts_by_class.get(cls,[])):
                if matched_flags[cls][i]:
                    continue
                iou = box_iou(p['box'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            per_class_scores[cls].append(p['score'])
            if best_iou >= iou_threshold:
                per_class_labels[cls].append(1)
                matched_flags[cls][best_idx] = True
            else:
                per_class_labels[cls].append(0)

    return per_class_scores, per_class_labels
