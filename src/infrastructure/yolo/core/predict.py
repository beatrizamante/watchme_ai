"""Module for YOLO object detection and prediction."""

from typing import Any, Dict, List, Union

import numpy as np

from config import YOLOSettings
from src.infrastructure.yolo.client.model import yolo_client
from src.infrastructure.yolo.scripts.get_bounding_boxes import \
    get_bounding_boxes

def predict(images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> List[Dict[str, Any]]:
    """
    Runs object detection and returns both bounding boxes and cropped person images.

    Args:
        images: Single image path/array or list of image paths/arrays to process.

    Returns:
        List of dictionaries containing detection results for each image.
        Returns empty list if no detections found.

    Raises:
        RuntimeError: If YOLO prediction fails.
        ValueError: If input images are invalid.
    """
    settings = YOLOSettings()
    model = yolo_client(settings.YOLO_MODEL_PATH)

    if not isinstance(images, list):
        images = [images]

    try:
        results = model.predict(
            images,
            stream=True,
            conf=0.28,
            classes=[0],
            verbose=False,
        )

        results_list = list(results)

        if not results_list:
            return []

        bounding_boxes = get_bounding_boxes(images, results_list)
        return bounding_boxes if bounding_boxes else []

    except Exception as e:
        raise RuntimeError(f"YOLO prediction failed: {str(e)}") from e
