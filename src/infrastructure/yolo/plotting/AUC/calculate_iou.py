"""
This module provides a function to calculate the Intersection over Union (IoU) metric
between two bounding boxes. The IoU is commonly used in object detection tasks to
measure the overlap between predicted and ground truth bounding boxes.
"""

def box_iou(box_a, box_b):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Args:
        box_a (list or tuple of float): The first bounding box in the format [x1, y1, x2, y2].
        box_b (list or tuple of float): The second bounding box in the format [x1, y1, x2, y2].
    Returns:
        float: The IoU value between box_a and box_b, ranging from 0.0 to 1.0.
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2]-box_a[0]) * max(0.0, box_a[3]-box_a[1])
    area_b = max(0.0, box_b[2]-box_b[0]) * max(0.0, box_b[3]-box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
