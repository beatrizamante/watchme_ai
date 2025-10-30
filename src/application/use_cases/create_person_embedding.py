from src._lib.encrypt import encrypt_embedding
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src.infrastructure.yolo.core.predict import predict

def create_person_embedding(file):
    """Embed a given user image
    Args:
        file: person image for embedding
    Raises:
        ValueErrorException: if no boxes are found
        Exception: in case the AI cannot process the frame
    Returns:
        embed: Embedded image if any are found
    """
    encode = OSNetEncoder()

    person_bbox_list = predict(file)

    if not person_bbox_list or not person_bbox_list[0]['detections']:
        raise ValueError("No person detected, please, try with another image")

    first_detection = person_bbox_list[0]['detections'][0]
    cropped_image = first_detection['cropped_image']

    try:
        encoding = encode.encode_single_image(cropped_image)
        if not encoding:
            raise ValueError("No person detected, please, try with another image")

        return encrypt_embedding(encoding)
    except Exception as e:
        raise e
