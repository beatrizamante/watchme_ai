from ultralytics import YOLO

from config import YOLOSettings
settings = YOLOSettings()

def yolo_client(weights):
    """
    Function returns a YOLO model
    Will load the weight if given.
    Or will return the base model.
    """
    return YOLO(weights)
