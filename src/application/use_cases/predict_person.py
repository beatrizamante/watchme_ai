import logging
import os
from src._lib.decrypt import decrypt_embedding
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict, predict_video, predict_single_frame
from src.infrastructure.yolo.scripts.convert_to_numpy_embedding import convert_to_numpy_embedding
from src.scripts.calculate_distance import compute_euclidean_distance

logger = logging.getLogger("watchmeai")
encoder = OSNetEncoder()

def predict_person_on_stream(chosen_person, stream):
    """
    Compare the chosen person's embedding to all detected people in the video stream.
    Args:
        chosen_person: Encrypted embedding of the target person.
        stream: Video file path, image path, or numpy array.
    Returns:
        List of matching bounding boxes.
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

    for frame_result in people_results:
        for detection in frame_result['detections']:
            all_cropped_images.append(detection['cropped_image'])
            all_bboxes.append(detection['bbox'])

    if not all_cropped_images:
        logger.info("No people detected")
        return []

    decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')
    encoded_batch = encoder.encode_batch(all_cropped_images)
    matches = []

    for i, encoded_person in enumerate(encoded_batch):
        distance = compute_euclidean_distance(decrypted_embedding, encoded_person)
        logger.debug(f"Distances {distance}")
        if distance < 0.3:
            matches.append({
                "bbox": all_bboxes[i],
                "distance": float(distance)
            })

    logger.info(f"Found {len(matches)} matches")
    return matches
