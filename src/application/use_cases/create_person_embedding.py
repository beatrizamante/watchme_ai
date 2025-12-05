import logging

from src._lib.encrypt import EncryptionService
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict

def create_person_embedding(
    file,
    encoder: OSNetEncoder,
    encryption_service: EncryptionService
):
    """Embed a given user image
    Args:
        file: person image for embedding (numpy array or file path)
        encoder: OSNet encoder service (optional, will use default if not provided)
        encryption_service: Encryption service (optional, will use default if not provided)
    Raises:
        ValueError: if no person is detected or encoding fails
        Exception: in case the AI cannot process the frame
    Returns:
        str: Encrypted embedding if successful (base64 encoded)
    """
    person_bbox_list = predict(file)

    if not person_bbox_list or not person_bbox_list[0]['detections']:
        raise ValueError("No person detected, please try with another image")

    first_detection = person_bbox_list[0]['detections'][0]
    cropped_image = first_detection['cropped_image']
    try:
        encoding = encoder.encode_single_image(cropped_image)

        if encoding is None or encoding.size == 0:
            raise ValueError("Failed to generate person embedding")

        encrypted_embedding = encryption_service.encrypt_embedding(encoding)

        return encrypted_embedding

    except Exception as e:
        logging.error("Error during encoding: %s", str(e))
        raise ValueError(f"Failed to create person embedding: {str(e)}") from e
