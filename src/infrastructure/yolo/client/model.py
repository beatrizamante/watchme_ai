from ultralytics import YOLO

from config import YOLOSettings
settings = YOLOSettings()

def yolo_client(weights=None):
    """
    Function returns a YOLO model
    Will load the weight if given.
    Or will return the base model.
    """
    model_path: str = weights if weights else settings.YOLO_MODEL_PATH
    return YOLO(model_path)
