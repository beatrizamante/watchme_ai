from typing import Any, Dict, List
import cv2

def get_bounding_boxes(
    images,
    results
) -> List[Dict[str, Any]] | None:
    """
    Extracts bounding boxes from YOLO detection results and crops corresponding regions from the input images.
    Args:
        images (list): List of image file paths or image arrays (numpy.ndarray).
        results (list): List of YOLO detection results, each containing bounding box information.
    Returns:
        list: A list of dictionaries, each containing:
            - 'detections': List of detections, where each detection is a dictionary with:
                - 'bbox': List of bounding box coordinates [x1, y1, x2, y2] as floats.
                - 'cropped_image': Cropped image region corresponding to the bounding box (numpy.ndarray).
                - 'crop_coordinates': List of bounding box coordinates [x1, y1, x2, y2] as integers.
            - 'original_image': The original image (numpy.ndarray).
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

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'cropped_image': cropped_person,
                        'crop_coordinates': [x1, y1, x2, y2]
                    }
                    detections.append(detection)

        processed_results.append({
            'detections': detections,
            'original_image': original_image
        })

        return processed_results
