from ultralytics import YOLO

trained_model = YOLO("../client/best.pt")
def predict(image):
    """
    Runs object detection prediction on the given image using the trained YOLO model.

    Args:
        image: The input image to perform prediction on. 
        Can be a file path, numpy array, or PIL Image depending on model requirements.

    Returns:
        results: The prediction results from the trained model, 
        typically including detected objects, bounding boxes, and confidence scores.
    """
    results = trained_model.predict(image)
    return results
