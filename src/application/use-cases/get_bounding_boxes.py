from src.infrastructure.yolo.core.predict import predict


def get_bounding_boxes(frame):
    """Get bounding boxes from frames fed to Yolo

    Args:
        frame (rgb): A given frame from video
    Raises:
        ValueErrorException: if no boxes are found
        Exception: in case the AI cannot process the frame 
    Returns:
        A list of bounding boxes
    """
    try:
        bounding_boxes = predict(frame)

        if not bounding_boxes:
            raise ValueError("No bounding boxes found.")
      
        return bounding_boxes
    except Exception as e:
        raise e

