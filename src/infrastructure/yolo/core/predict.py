from ultralytics import YOLO

trained_model = YOLO("../best_model")
def predict(image):
    results = trained_model.predict(image)
    return results
