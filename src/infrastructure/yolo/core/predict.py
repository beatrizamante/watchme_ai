"""Module for YOLO object detection and prediction."""

from typing import Any, Dict, List, Union

import numpy as np
from ultralytics import YOLO

from src.infrastructure.yolo.scripts.get_bounding_boxes import \
    get_bounding_boxes


def load_trained_model(model_path: str | None = None) -> YOLO:
    """Load the trained YOLO model."""
    if model_path is None:
        model_path = "src/infrastructure/yolo/client/best.pt"
    return YOLO(model_path)


trained_model = load_trained_model()


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
    if not isinstance(images, list):
        images = [images]

    try:
        results = trained_model.predict(
            images,
            stream=True,
            conf=0.3,
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
