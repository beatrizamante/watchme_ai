"""Model tuner for finding best parameters"""

from pathlib import Path

import ray
from ray import tune
from ray.tune.error import TuneError

from config import YOLOSettings
from src.infrastructure.yolo.client.model import yolo_client


class HyperparameterTuner:
    """Handle hyperparameter tuning with Ray Tune (with resume support)"""
    def __init__(self):
        self.settings = YOLOSettings()
        self.best_params = None
        self.model = None

    def _get_search_space(self):
        """Define hyperparameter search space for YOLO training"""
        return {
            "lr0": tune.uniform(1e-5, 1e-2),
            "momentum": tune.uniform(0.6, 0.98),
            "box": tune.uniform(0.02, 0.2),
            "cls": tune.uniform(0.1, 2.0),
            "hsv_s": tune.uniform(0.0, 0.9),
            "hsv_v": tune.uniform(0.0, 0.9),
            "degrees": tune.uniform(0.0, 45.0),
            "translate": tune.uniform(0.0, 0.9),
            "scale": tune.uniform(0.0, 0.9),
            "shear": tune.uniform(0.0, 10.0),
            "dropout": tune.uniform(0.0, 0.3),
        }

    def _check_for_checkpoint(self):
        """Check if there's an existing ray_tune checkpoint to resume from"""
        ray_tune_dir = Path(self.settings.YOLO_PROJECT_PATH) / "ray_tune"
        if ray_tune_dir.exists() and any(ray_tune_dir.iterdir()):
            return True
        return False

    def tune(self, baseline_weights=None, iterations=5, epochs=20):
        """
        Tune hyperparameters using Ray Tune with resume support

        Args:
            baseline_weights (str, optional): Path to baseline weights file (.pt)
            iterations (int): Number of tuning trials to run
            epochs (int): Number of epochs per trial

        Returns:
            ray.tune.ExperimentAnalysis: Results object containing best hyperparameters

        Raises:
            RuntimeError: If Ray Tune encounters an error
            TuneError: If there's an issue with the tuning process
        """
        try:
            if ray.is_initialized():
                ray.shutdown()

            ray.init(
                ignore_reinit_error=True,
                num_cpus=8,
                num_gpus=1 if self.settings.YOLO_DEVICE != "cpu" else 0
            )

            self.model = yolo_client(baseline_weights)
            resume_checkpoint = self._check_for_checkpoint()
            search_space = self._get_search_space()

            results = self.model.tune(
                data=self.settings.YOLO_DATASET_PATH,
                use_ray=True,
                space=search_space,
                epochs=epochs,
                iterations=iterations,
                grace_period=10,
                gpu_per_trial=1 if self.settings.YOLO_DEVICE != 'cpu' else 0,
                project=self.settings.YOLO_PROJECT_PATH,
                name="ray_tune",
                resume=resume_checkpoint
            )

            self.best_params = results.get_results() # type: ignore
            return results

        except (RuntimeError, TuneError) as e:
            print(f"Error during model tuning: {e}")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()

    def get_best_params(self):
        """Get the best hyperparameters from the last tuning run"""
        return self.best_params
