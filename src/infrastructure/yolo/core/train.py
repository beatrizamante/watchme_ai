from config import settings
from ..client.client import yolo_client

def train():
    results = yolo_client.train(
        data="src/dataset/yolo/dataset.yml",
        epochs=settings.YOLO_EPOCHS,
        batch=settings.YOLO_BATCH_SIZE,
        optimizer=settings.YOLO_LOSS_FUNC,
        lr0=settings.YOLO_LEARNING_RATE,
        dropout=settings.YOLO_DROPOUT,
        imgsz=640,
        device=settings.YOLO_DEVICE        
        )
       
    return results

if __name__ == "__main__":
    try:
        r = train()
        print("✅ Funcionou!")
        print(r)
    except Exception as e:
        print("❌ Deu erro:")
        print(e)  