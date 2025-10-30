import os

from config import YOLOSettings
from src.infrastructure.yolo.client.model import yolo_client


class YOLOTrainer:
    """Handle YOLO training operations"""

    def __init__(self):
        self.settings = YOLOSettings()
        self.model = None
        self.results = None

    def load_model(self, weights=None):
        """Load YOLO model"""
        self.model = yolo_client(weights)
        print(f"✓ Model loaded: {getattr(self.model, 'baseline_weights', 'Unknown path')}")
        return self.model

    def train(self, weights=None, hyperparams=None):
        """
        Trains the YOLO model using the specified weights and hyperparameters.
        Args:
            weights (str or None): Path to the pretrained weights file. If None, uses default weights.
            hyperparams (dict or None): Dictionary of hyperparameters to override defaults. Supported keys include:
                - "box" (int): Box loss gain.
                - "cls" (float): Class loss gain.
                - "lr0" (float): Initial learning rate.
                - "dropout" (float): Dropout rate.
        Raises:
            ValueError: If the YOLO model fails to load or does not have a 'train' method.
        Returns:
            Any: Training results returned by the YOLO model's train method.
        """
        self.load_model(weights)
        if self.model is None or not hasattr(self.model, "train"):
            raise ValueError("Failed to load YOLO model or 'train' method not found.")

        hp = hyperparams if hyperparams else {}
        box = hp.get("box", 8)
        cls = hp.get("cls", 0.5)
        lr0 = hp.get("lr0", self.settings.YOLO_LEARNING_RATE)
        dropout = hp.get("dropout", self.settings.YOLO_DROPOUT)

        self.results = self.model.train(
            data=self.settings.YOLO_DATASET_PATH,
            project=self.settings.YOLO_PROJECT_PATH,
            name="baseline_train",
            multi_scale=True,
            amp=True,
            freeze=5,
            box=box,
            cls=cls,
            epochs=self.settings.YOLO_EPOCHS,
            batch=self.settings.YOLO_BATCH_SIZE,
            optimizer=self.settings.YOLO_LOSS_FUNC,
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
