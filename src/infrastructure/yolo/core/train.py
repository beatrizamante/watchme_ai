from config import settings
from ..client.client import yolo_client

##Kerasturner

def train():
    results = yolo_client.train(
        data="src/dataset/yolo/dataset.yaml",
        epochs=settings.YOLO_EPOCHS,
        imgsz=settings.YOLO_IMAGE_SIZE
    )
    return results
