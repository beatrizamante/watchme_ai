from config import settings
from ..client.model import yolo_client


def train(weights=None, hp=None):
    epochs = hp.get("epochs", settings.YOLO_EPOCHS) if hp else settings.YOLO_EPOCHS
    batch = hp.get("batch", settings.YOLO_BATCH_SIZE) if hp else settings.YOLO_BATCH_SIZE
    optimizer = hp.get("optimizer", settings.YOLO_LOSS_FUNC) if hp else settings.YOLO_LOSS_FUNC
    lr0 = hp.get("lr0", settings.YOLO_LEARNING_RATE) if hp else settings.YOLO_LEARNING_RATE
    dropout = hp.get("dropout", settings.YOLO_DROPOUT) if hp else settings.YOLO_DROPOUT

    model = yolo_client(weights if weights else settings.YOLO_MODEL_PATH)
    
    results = model.train(
        data="src/dataset/yolo/dataset.yml",
        epochs=epochs,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        dropout=dropout,
        imgsz=640,
        device=settings.YOLO_DEVICE
        )
    return results