import os

from src.infrastructure.yolo.client.model import yolo_client


class YOLOTrainer:
    """Handle YOLO training operations"""

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.results = None

    def load_model(self, weights=None):
        """Load YOLO model"""
        self.model = yolo_client(weights)
        print(f"✓ Model loaded: {getattr(self.model, 'baseline_weights', 'Unknown path')}")
        return self.model

    def train(self, weights=None, hyperparams=None):
        """
        Train the YOLO model with specified or default hyperparameters.
        This method loads a YOLO model and initiates training with configurable
        hyperparameters. If no hyperparameters are provided, default values from
        settings are used.
        Args:
            weights (str, optional): Path to pre-trained weights file. If None,
                uses default model loading behavior.
            hyperparams (dict, optional): Dictionary containing training hyperparameters.
                Supported keys:
                - epochs (int): Number of training epochs
                - batch (int): Batch size for training
                - optimizer (str): Optimizer type
                - lr0 (float): Initial learning rate
                - dropout (float): Dropout rate
        Returns:
            object: Training results object containing metrics, logs, and model
                performance data from the completed training session.
        Raises:
            ValueError: If the model fails to load or doesn't have a 'train' method.
        """

        self.load_model(weights)
        if self.model is None or not hasattr(self.model, "train"):
            raise ValueError("Failed to load YOLO model or 'train' method not found.")

        hp = hyperparams if hyperparams else {}
        epochs = hp.get("epochs", self.settings.YOLO_EPOCHS)
        batch = hp.get("batch", self.settings.YOLO_BATCH_SIZE)
        optimizer = hp.get("optimizer", self.settings.YOLO_LOSS_FUNC)
        lr0 = hp.get("lr0", self.settings.YOLO_LEARNING_RATE)
        dropout = hp.get("dropout", self.settings.YOLO_DROPOUT)

        print(f"Training with: epochs={epochs}, batch={batch}, lr0={lr0}")

        self.results = self.model.train(
            data=self.settings.YOLO_DATASET_PATH,
            project=self.settings.YOLO_PROJECT_PATH,
            name="baseline_train",
            multi_scale=True,
            amp=True,
            freeze=5,
            box=8,
            epochs=epochs,
            batch=batch,
            optimizer=optimizer,
            lr0=lr0,
            dropout=dropout,
            imgsz=640,
            device=self.settings.YOLO_DEVICE,
        )

        return self.results

    def get_best_weights_path(self):
        """Get path to best weights from last training"""
        if self.results is None:
            raise ValueError("No training results available. Train model first.")
        return os.path.join(str(self.results.save_dir), "weights", "best.pt")
