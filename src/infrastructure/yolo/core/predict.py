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
    Runs object detection on video frames with timestamp information.
    """
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        raise ValueError(f"Unsupported video format: {video_path}")

    import os
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {video_width}x{video_height}, {fps:.2f} FPS, {total_frames} frames")

    results = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                try:
                    actual_h, actual_w = frame.shape[:2]

                    if actual_w != video_width or actual_h != video_height:
                        print(f"Frame size mismatch! Video: {video_width}x{video_height}, Frame: {actual_w}x{actual_h}")

                    frame_results = model.predict(
                        frame,
                        conf=0.28,
                        classes=[0],
                        verbose=False
                    )

                    frame_info = {
                        'frame_number': int(frame_count),              # Python int
                        'timestamp': float(frame_count / fps),         # Python float
                        'original_video_size': [int(video_width), int(video_height)],  # Python ints
                        'processed_frame_size': [int(actual_w), int(actual_h)],        # Python ints
                        'scale_factor_x': float(actual_w / video_width),               # Python float
                        'scale_factor_y': float(actual_h / video_height)               # Python float
                    }

                    if hasattr(frame_results, '__iter__') and not isinstance(frame_results, (str, bytes)):
                        results_list = list(frame_results)
                    else:
                        results_list = [frame_results]

                    frame_detections = get_bounding_boxes([frame], results_list, frame_info)

                    if frame_detections:
                        for frame_detection in frame_detections:
                            frame_detection['video_metadata'] = {
                                'original_video_size': [video_width, video_height],
                                'fps': fps,
                                'total_frames': total_frames,
                                'video_path': video_path
                            }
                        results.extend(frame_detections)

                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    continue

            frame_count += 1

    finally:
        cap.release()

    return results

def predict_single_frame(frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> List[Dict[str, Any]]:
    """
    Runs object detection on a single frame (for WebSocket streaming).

    Args:
        frame: OpenCV image array (numpy.ndarray)
        frame_number: Frame number for tracking
        timestamp: Timestamp in seconds
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

        frame_info = {
            'frame_number': frame_number,
            'timestamp': timestamp
        }

        frame_detections = get_bounding_boxes([frame], [results], frame_info)
        return frame_detections if frame_detections else []

    except Exception as e:
        raise RuntimeError(f"Frame prediction failed: {str(e)}") from e
