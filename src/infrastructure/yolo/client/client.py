from ultralytics import YOLO
from src.config import settings

yolo_client = YOLO(settings.YOLO_MODEL_PATH)
