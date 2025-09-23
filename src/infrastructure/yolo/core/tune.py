import keras_tuner as kt
from ultralytics import YOLO
from config import settings

yolo_client = YOLO("runs/train/exp/weights/best.pt")

def tune(hp):
    epochs = hp.Int("epochs", min_value=50, max_value=150, step=10)
    batch = hp.Choice("batch", values=[16, 32, 64])
    lr0 = hp.Float("lr0", 1e-5, 1e-2, sampling="log")
    dropout = hp.Float("dropout", 0.0, 0.3, step=0.1)
    optimizer = hp.Choice("optimizer", ["Adam", "SGD"])

    results = yolo_client.train(
        data="src/dataset/yolo/dataset.yml",
        epochs=epochs,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        dropout=dropout,
        imgsz=640,
        device=settings.YOLO_DEVICE,
        verbose=False
    )

    metrics = results.results_dict
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    return map50_95


if __name__ == "__main__":
    tuner = kt.Hyperband(
        tune,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_epochs=100,
        factor=3,
        directory="kt_tuning",
        project_name="yolo_person"
    )

    tuner.search()
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("epochs:", best_hp.get("epochs"))
    print("batch:", best_hp.get("batch"))
    print("lr0:", best_hp.get("lr0"))
    print("dropout:", best_hp.get("dropout"))
    print("optimizer:", best_hp.get("optimizer"))
