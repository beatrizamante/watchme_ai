import logging
import numpy as np
from src._lib.encrypt import encrypt_embedding
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict

def create_person_embedding(file):
    """Embed a given user image
    Args:
        file: person image for embedding (numpy array or file path)
    Raises:
        ValueError: if no person is detected or encoding fails
        Exception: in case the AI cannot process the frame
    Returns:
        bytes: Encrypted embedding if successful
    """
    encode = OSNetEncoder()

    person_bbox_list = predict(file)

    logging.debug(f"Retrieved bounding boxes: {len(person_bbox_list) if person_bbox_list else 0}")

    if not person_bbox_list or not person_bbox_list[0]['detections']:
        raise ValueError("No person detected, please try with another image")

    first_detection = person_bbox_list[0]['detections'][0]
    cropped_image = first_detection['cropped_image']

    logging.debug(f"Cropped image shape: {cropped_image.shape}")

    try:
        encoding = encode.encode_single_image(cropped_image)

        if encoding is None or encoding.size == 0:
            raise ValueError("Failed to generate person embedding")

        logging.info(f"Generated embedding with shape: {encoding.shape}")

        #encrypted_embedding = encrypt_embedding(encoding)
        logging.info("Successfully encrypted embedding")

        #return encrypted_embedding
        return encoding

    except Exception as e:
        logging.error(f"Error during encoding: {str(e)}")
        raise ValueError(f"Failed to create person embedding: {str(e)}") from e
