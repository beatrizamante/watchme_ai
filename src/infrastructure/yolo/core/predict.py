from infrastructure.yolo.client import yolo_client

def predict(image_path: str):
    results = yolo_client.predict(image_path)
    return results
