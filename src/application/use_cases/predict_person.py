import logging
import os
import base64

import numpy as np
from src._lib.decrypt import decrypt_embedding
from src.infrastructure.osnet.core.encode import get_encoder
from src.infrastructure.yolo.core.predict import predict, predict_video, predict_single_frame
from src.scripts.calculate_distance import compute_batch_distances, compute_euclidean_distance

logger = logging.getLogger("watchmeai")
encoder = get_encoder()

def predict_person_on_stream(chosen_person, stream):
    """
    Compare the chosen person's embedding to all detected people in the video stream.
    Args:
        chosen_person: Encrypted embedding of the target person.
        stream: Video file path, image path, or numpy array.
    Returns:
        List of matching bounding boxes with timestamps.
    """
    logger.info(f"Processing stream: {type(stream)} - {stream if isinstance(stream, str) else 'numpy array'}")

    if isinstance(stream, str):
        if os.path.exists(stream):
            if stream.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                logger.info("Processing video file")
                people_results = predict_video(stream)
            elif stream.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                logger.info("Processing image file")
                people_results = predict(stream)
            else:
                raise ValueError(f"Unsupported file format: {stream}")
        else:
            raise FileNotFoundError(f"File not found: {stream}")
    else:
        logger.info("Processing single frame")
        people_results = predict_single_frame(stream)

    all_cropped_images = []
    all_bboxes = []
    all_frame_info = []

    for frame_result in people_results:
        for detection in frame_result['detections']:
            all_cropped_images.append(detection['cropped_image'])
            all_bboxes.append(detection['bbox'])

            frame_info = {
                'frame_number': detection.get('frame_number', 0),
                'timestamp': detection.get('timestamp', 0.0),
                'coordinate_system': detection.get('coordinate_system', {}),
                'crop_info': detection.get('crop_info', {}),
                'confidence': detection.get('confidence', 1.0),
                'detection_id': detection.get('detection_id', 0)
            }
            all_frame_info.append(frame_info)

    if not all_cropped_images:
        logger.info("No people detected")
        return []

    decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')
    encoded_batch = encoder.encode_batch(all_cropped_images)
    matches = []

    if decrypted_embedding.shape != (512,):
        logger.error(f"Invalid decrypted embedding shape: {decrypted_embedding.shape}")
        return []

    if not encoded_batch:
        logger.warning("OSNet encoding returned empty batch")
        return []

    if len(encoded_batch) != len(all_cropped_images):
        logger.error(f"Batch size mismatch: {len(encoded_batch)} vs {len(all_cropped_images)}")
        return []

    distances = compute_batch_distances(decrypted_embedding, np.array(encoded_batch))

    for i, distance in enumerate(distances):
        distance = float(distance)
        logger.debug(f"Distance: {distance} at frame {all_frame_info[i]['frame_number']}")

        if distance < 0.6:
            match = {
                "bbox": all_bboxes[i],
                "bbox_format": "xyxy",
                "coordinate_info": all_frame_info[i]['coordinate_system'],
                "crop_info": all_frame_info[i]['crop_info'],
                "confidence": all_frame_info[i]['confidence'],
                "distance": distance,
                "frame_number": all_frame_info[i]['frame_number'],
                "timestamp": all_frame_info[i]['timestamp'],
                "time_formatted": format_timestamp(all_frame_info[i]['timestamp'])
            }
            matches.append(match)

    return matches

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
