import cv2
from src._lib.decrypt import decrypt_embedding
from src.application.use_cases.timestamp import calculate_timestamp
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict
from src.scripts.calculate_distance import compute_euclidean_distance

encoder = OSNetEncoder()

def predict_person_on_stream(chosen_person, stream):
    """
    Compare the chosen person's embedding to all detected people in the video stream.
    Args:
        chosen_person: Encrypted embedding of the target person.
        stream: Video frame(s) or stream to process (file path or numpy array).
    Returns:
        List of matching bounding boxes with timestamps.
    """
    video_info = None
    if isinstance(stream, str):
        video_info = get_video_info(stream)

    people_results = predict(stream)
    all_cropped_images = []
    all_bboxes = []
    frame_indices = []

    for frame_idx, frame_result in enumerate(people_results):
        for detection in frame_result['detections']:
            all_cropped_images.append(detection['cropped_image'])
            all_bboxes.append(detection['bbox'])
            frame_indices.append(frame_idx)

    if not all_cropped_images:
        return []

    decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')
    encoded_batch = encoder.encode_batch(all_cropped_images)
    matches = []

    for i, encoded_person in enumerate(encoded_batch):
        distance = compute_euclidean_distance(decrypted_embedding, encoded_person)
        if distance < 0.5:

            timestamp = calculate_timestamp(frame_indices[i], video_info)

            matches.append({
                "bbox": all_bboxes[i],
                "distance": float(distance),
                "timestamp": timestamp,
                "confidence": 1.0 - min(distance, 1.0)
            })

    return matches

def get_video_info(video_path):
    """Extract video metadata for timestamp calculation"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "path": video_path
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None
