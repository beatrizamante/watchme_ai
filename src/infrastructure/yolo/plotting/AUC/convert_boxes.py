"""
This module provides utility functions for working with bounding boxes in object detection tasks,
specifically for converting YOLO format bounding boxes to pixel coordinates.
"""

def yolo_xywh_to_xyxy(box, img_w, img_h):
    """
    Convert YOLO format bounding box (x_center, y_center, width, height) to (x1, y1, x2, y2) format.

    Args:
        box (list or tuple): Bounding box in YOLO format [x_center, y_center, width, height], normalized (values between 0 and 1).
        img_w (int or float): Image width in pixels.
        img_h (int or float): Image height in pixels.

    Returns:
        list: Bounding box in [x1, y1, x2, y2] format, in pixel coordinates.
    """
    x_c, y_c, w, h = box
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h
    x1 = x_c - w/2
    y1 = y_c - h/2
    x2 = x_c + w/2
    y2 = y_c + h/2
    return [x1, y1, x2, y2]
