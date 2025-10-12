import os
from pathlib import Path
from config import Settings
from src.infrastructure.yolo.core.train import YOLOTrainer
from src.infrastructure.yolo.core.tune import HyperparameterTuner
class YOLOPipeline:
    """Complete YOLO training pipeline with resume support"""

    def __init__(self, settings=None):
        self.settings = settings if settings else Settings(
            YOLO_MODEL_PATH="src/infrastructure/yolo/client/best.pt",
            YOLO_EPOCHS=70,
            YOLO_BATCH_SIZE=16,
            YOLO_LEARNING_RATE=0.001,
            YOLO_LOSS_FUNC="AdamW",
            YOLO_DROPOUT=0.0,
            YOLO_DEVICE=0,
            YOLO_DATASET_PATH="src/dataset/yolo/dataset.yml",
            YOLO_PROJECT_PATH="src/dataset/yolo/runs/detect",
        )
        
        self.trainer = YOLOTrainer(self.settings)
        self.tuner = HyperparameterTuner(self.settings)
        self.baseline_weights = None
        self.best_hyperparams = None
        self.final_results = None
        
    def _check_for_baseline_weights(self):
        """Check if there's an existing baseline weights to resume from"""
        baseline_dir = Path(self.settings.YOLO_PROJECT_PATH) / "baseline_train"
        best_weights_path = baseline_dir / "weights" / "best.pt"
        
        if best_weights_path.exists():
            return True, str(best_weights_path)
        return False, None

    def run(self):
        """
        Run complete pipeline: baseline training -> tuning -> final training
        """
        print("="*60)
        print("YOLO Training & Hyperparameter Tuning Pipeline")
        print("="*60)

        has_baseline, baseline_path = self._check_for_baseline_weights()
        
        if not has_baseline:
            print("\n[1/3] Training baseline model...")
            print("-" * 60)
            
            baseline_results = self.trainer.train()
            
            if baseline_results and hasattr(baseline_results, 'save_dir'):
                self.baseline_weights = self.trainer.get_best_weights_path()
                print(f"✓ Baseline training completed: {self.baseline_weights}")
            else:
                raise RuntimeError("Training failed or returned invalid results")
        else:
            print(f"\n[1/3] Found existing baseline weights: {baseline_path}")
            self.baseline_weights = baseline_path

        print("\n[2/3] Running hyperparameter tuning...")
        print("-" * 60)
        tune_results = self.tuner.tune(
            baseline_weights=self.baseline_weights,
        )
        
        self.best_hyperparams = tune_results.get_best_result().config # type: ignore
        print(f"✓ Best hyperparameters found: {self.best_hyperparams}")

        print("\n[3/3] Final training with optimized hyperparameters...")
        print("-" * 60)
        self.final_results = self.trainer.train(
            weights=self.baseline_weights,
            hyperparams=self.best_hyperparams
        )

        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print(f"Baseline weights: {self.baseline_weights}")
        print(f"Final model: {self.final_results.save_dir}") # type: ignore
        print("="*60)

        return self.final_results