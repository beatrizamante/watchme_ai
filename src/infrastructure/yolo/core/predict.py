"""Module for YOLO object detection and prediction."""

from typing import Any, Dict, List, Union

import cv2
import numpy as np

from config import YOLOSettings
from src.infrastructure.yolo.client.model import yolo_client
from src.infrastructure.yolo.scripts.get_bounding_boxes import \
    get_bounding_boxes

settings = YOLOSettings()
model = yolo_client(settings.YOLO_MODEL_PATH)

def predict(images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> List[Dict[str, Any]]:
    """
    Runs object detection on images.

    Args:
        images: Single image path/array or list of image paths/arrays to process.

    Returns:
        List of dictionaries containing detection results for each image.
    """

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

def predict_video(video_path: str, frame_skip: int = 30) -> List[Dict[str, Any]]:
    """
    Runs object detection on video frames.

    Args:
        video_path: Path to the video file
        frame_skip: Process every Nth frame (default: 30 = ~1 FPS for 30fps video)

    Returns:
        List of dictionaries containing detection results for each processed frame.
    """
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        raise ValueError(f"Unsupported video format: {video_path}")

    import os
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    results = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                try:
                    # Get YOLO results for single frame
                    frame_results = model.predict(
                        frame,
                        conf=0.28,
                        classes=[0],
                        verbose=False
                    )

                    frame_detections = get_bounding_boxes([frame], frame_results)

                    if frame_detections:
                        for detection in frame_detections:
                            detection['frame_number'] = frame_count
                            detection['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                        results.extend(frame_detections)

                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    continue

            frame_count += 1

    finally:
        cap.release()

    return results

def predict_single_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Runs object detection on a single frame (for WebSocket streaming).

    Args:
        frame: OpenCV image array (numpy.ndarray)

    Returns:
        List of dictionaries containing detection results.
    """

    try:
        results = model.predict(
            frame,
            conf=0.28,
            classes=[0],
            verbose=False
        )

        frame_detections = get_bounding_boxes([frame], [results])
        return frame_detections if frame_detections else []

    except Exception as e:
        raise RuntimeError(f"Frame prediction failed: {str(e)}") from e
