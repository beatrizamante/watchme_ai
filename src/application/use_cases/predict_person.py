from src._lib.decrypt import decrypt_embedding
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict
from src.scripts.calculate_distance import calculate_distance

encoder = OSNetEncoder()

def predict_person_on_stream(chosen_person, stream):
    """
    Compare the chosen person's embedding to all detected people in the video stream.
    Args:
        chosen_person: Encrypted embedding of the target person.
        stream: Video frame(s) or stream to process.
    Returns:
        List of matching bounding boxes.
    """
    people_results = predict(stream)
    all_cropped_images = []
    all_bboxes = []

    for frame_result in people_results:
        for detection in frame_result['detections']:
            all_cropped_images.append(detection['cropped_image'])
            all_bboxes.append(detection['bbox'])

    if not all_cropped_images:
        return []

    decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')

    encoded_batch = encoder.encode_batch(all_cropped_images)
    matches = []

    for i, encoded_person in enumerate(encoded_batch):
        distance = calculate_distance(decrypted_embedding, encoded_person)
        if distance < 0.8:
            matches.append({
                "bbox": all_bboxes[i],
                "distance": distance
            })

    return matches
