from ultralytics import YOLO
from config import settings

yolo_client = YOLO(settings.YOLO_MODEL_PATH)
