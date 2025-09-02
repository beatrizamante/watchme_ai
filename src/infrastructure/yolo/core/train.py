from infrastructure.yolo.client import yolo_client
from config import settings

def train():
    results = yolo_client.train(
        data="dataset.yaml",
        epochs=settings.YOLO_EPOCHS,
        imgsz=settings.YOLO_IMAGE_SIZE
    )
    return results
