from typing import Any, Dict, List, Union

import cv2

def get_bounding_boxes(
    images,
    results,
    frame_info=None
) -> Union[List[Dict[str, Any]], None]:
    """
    Extracts bounding boxes from YOLO detection results and crops corresponding regions from the input images.
    Args:
        images (list): List of image file paths or image arrays (numpy.ndarray).
        results (list): List of YOLO detection results, each containing bounding box information.
        frame_info (dict, optional): Dictionary with frame metadata like frame_number and timestamp.
    Returns:
        list: A list of dictionaries, each containing detection data with optional timestamp info.
    """
    processed_results = []

    for i, result in enumerate(results):
        if isinstance(images[i], str):
            original_image = cv2.imread(images[i])
        else:
            original_image = images[i]

        if original_image is None:
            continue

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                h, w = original_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    cropped_person = original_image[y1:y2, x1:x2]

                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    x = float(x1)
                    y = float(y1)
                    detection = {
                        'bbox': [x, y, w, h],
                        'cropped_image': cropped_person,
                    }

                    if frame_info:
                        detection['frame_number'] = frame_info.get('frame_number', 0)
                        detection['timestamp'] = frame_info.get('timestamp', 0.0)

                    detections.append(detection)

        frame_data = {
            'detections': detections,
            'original_image': original_image
        }

        if frame_info:
            frame_data['frame_number'] = frame_info.get('frame_number', 0)
            frame_data['timestamp'] = frame_info.get('timestamp', 0.0)

        processed_results.append(frame_data)

    return processed_results
