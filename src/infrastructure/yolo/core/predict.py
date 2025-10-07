from ultralytics import YOLO

trained_model = YOLO("../best_model")
def predict(image_path: str):
    results = trained_model.predict(image_path)
    return results
