import keras_tuner as kt
from .train import train

def tune(weights=None):
    def model_builder(hp):
        hyperparams = {
            "epochs": hp.Int("epochs", min_value=50, max_value=150, step=10),
            "batch": hp.Choice("batch", values=[16, 32, 64]),
            "lr0": hp.Float("lr0", 1e-5, 1e-2, sampling="log"),
            "dropout": hp.Float("dropout", 0.0, 0.3, step=0.1),
            "optimizer": hp.Choice("optimizer", ["Adam", "SGD"])
        }
        
        results = train(weights=weights, hp=hyperparams)
        
        metrics = getattr(results, "results_dict", {})
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        return map50_95

    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("map50_95", direction="max"),
        max_epochs=100,
        factor=3,
        directory="kt_tuning",
        project_name="yolo_person"
    )

    tuner.search()
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp


