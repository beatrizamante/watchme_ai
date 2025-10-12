"""Model tuner for finding best parameters
"""

import ray
from ray import tune
from ray.tune.error import TuneError

from config import settings

from ..client.model import yolo_client


def model_tune(baseline_weights=None):
    """The function for model tuning after the first baseline training.
    Args:
        baseline_weights (PyTorch, optional): The baseline weights of the first training. Defaults to None.
    Returns:
        engine.Results: The dictionary of tuning results
        PyTorchTensor: Saves the .pt best set of weights
    """
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
    ignore_reinit_error=True,
    num_cpus=8, #For google colab
    num_gpus=1 if settings.YOLO_DEVICE != 'cpu' else 0
    )

    model = yolo_client(baseline_weights if baseline_weights else settings.YOLO_MODEL_PATH)

    search_space = {
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

    try:
        results = model.tune(
            data="src/dataset/yolo/dataset.yml",
            use_ray=True,
            space=search_space,
            epochs=20,
            iterations=5,
            grace_period=10,
            gpu_per_trial=1 if settings.YOLO_DEVICE != 'cpu' else 0,
            project="src/runs/detect",
            name="ray_tune",
        )  
        return results
    except (RuntimeError, TuneError) as e:
        print(f"Error during model tuning: {e}")
        raise e
    finally:
        ray.shutdown()
