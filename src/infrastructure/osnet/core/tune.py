import keras_tuner as kt
from .train import train

def tune(dataset_path):
    """
    Hyperparameter tuning for OSNet using Keras Tuner
    
    Args:
        dataset_path: Path to the dataset
    
    Returns:
        best_hp: Best hyperparameters found
    """
    def model_builder(hp):
        # Define hyperparameter search space for OSNet
        hyperparams = {
            "max_epoch": hp.Int("max_epoch", min_value=30, max_value=100, step=10),
            "lr": hp.Float("lr", 1e-5, 1e-2, sampling="log"),
            "weight_decay": hp.Float("weight_decay", 1e-6, 1e-3, sampling="log"),
            "batch_size": hp.Choice("batch_size", values=[32, 64, 128]),
            "optimizer": hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])
        }
        
        # Use your existing train function
        results = train(dataset_path=dataset_path, hp=hyperparams)
        
        # Return the metric you want to optimize (Rank-1 accuracy is common for ReID)
        rank1_accuracy = results.get("rank1", 0.0)
        return rank1_accuracy

    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("rank1_accuracy", direction="max"),
        max_epochs=50,
        factor=3,
        directory="kt_tuning_osnet",
        project_name="osnet_reid"
    )

    tuner.search()
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hp