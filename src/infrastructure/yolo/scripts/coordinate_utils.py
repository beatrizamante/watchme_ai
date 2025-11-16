# src/utils/coordinate_utils.py
def convert_coordinates_for_frontend(bbox, source_size, target_size, bbox_format='xyxy'):
    """
    Convert bounding box coordinates between different image sizes.

    Args:
        bbox: Bounding box coordinates
        source_size: [width, height] of source image
        target_size: [width, height] of target image
        bbox_format: 'xyxy' or 'xywh'

    Returns:
        Converted coordinates in same format
    """
    source_w, source_h = source_size
    target_w, target_h = target_size

    scale_x = target_w / source_w
    scale_y = target_h / source_h

    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        return [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ]
    elif bbox_format == 'xywh':
        x, y, w, h = bbox
        return [
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y
        ]

def bbox_xyxy_to_xywh(bbox):
    """Convert [x1,y1,x2,y2] to [x,y,w,h]"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]

def bbox_xywh_to_xyxy(bbox):
    """Convert [x,y,w,h] to [x1,y1,x2,y2]"""
    x, y, w, h = bbox
    return [x, y, x+w, y+h]
