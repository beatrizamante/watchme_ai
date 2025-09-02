from src.infrastructure.yolo.client.client import yolo_client
from config import settings

##Kerasturner

def train():
    results = yolo_client.train(
        data="src/dataset/yolo/dataset.yaml",
        epochs=settings.YOLO_EPOCHS,
        imgsz=settings.YOLO_IMAGE_SIZE
    )
    return results
