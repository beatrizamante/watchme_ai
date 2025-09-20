from config import settings
from ..client.client import yolo_client

def train():
    results = yolo_client.train(
        data="src/dataset/yolo/dataset.yaml",
        epochs=settings.YOLO_EPOCHS,
        batch=settings.YOLO_IMAGE_SIZE,
        optimizer=settings.YOLO_LOSS_FUNC,
        lr0=settings.YOLO_LEARNING_RATE,
        dropout=settings.YOLO_DROPOUT,
        imgsz=640   
        )
       
    return results
