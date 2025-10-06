import os
import json
import uuid
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
from config import settings
from ..client.model import yolo_client 

def _extract_map50_95(results, model=None, data=None):
    try:
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "mAP50-95", "map50-95"):
                if k in results.results_dict:
                    return float(results.results_dict[k])
    except Exception:
        pass

    try:
        if hasattr(results, "metrics") and isinstance(results.metrics, dict):
            for k in ("map50_95", "mAP50-95", "map"):
                if k in results.metrics:
                    return float(results.metrics[k])
    except Exception:
        pass

    try:
        if model is not None and data is not None:
            val_res = model.val(data=data)
            if hasattr(val_res, "box") and hasattr(val_res.box, "map"):
                return float(val_res.box.map)
    except Exception:
        pass

    return 0.0


def train_yolo_ray(config, baseline_weights=None, data="src/dataset/yolo/dataset.yml",
                   project_root="src/runs/detect"):
    run_name = f"trial_{uuid.uuid4().hex[:8]}"

    model = yolo_client(baseline_weights) if baseline_weights else yolo_client(None)

    results = model.train(
        data = data,
        epochs = int(config.get("epochs", 50)),
        batch = int(config.get("batch", 16)),
        lr0 = float(config.get("lr0", 1e-3)),
        optimizer = config.get("optimizer", "Adam"),
        dropout = float(config.get("dropout", 0.0)),
        imgsz = int(config.get("imgsz", 640)),
        device = settings.YOLO_DEVICE,
        project = project_root,
        name = run_name,
        exist_ok = True,  
        resume = False,   
        verbose = False
    )

    run_dir = os.path.join(project_root, run_name)
    weights_dir = os.path.join(run_dir, "weights")
    best_weights = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_weights):
        last = os.path.join(weights_dir, "last.pt")
        best_weights = last if os.path.exists(last) else None

    map50_95 = _extract_map50_95(results, model=model, data=data)

    meta = {
        "hp": config,
        "map50_95": float(map50_95),
        "weights_path": best_weights,
        "run_dir": run_dir
    }
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # report to Ray Tune
    tune.report(map50_95=float(map50_95))


def tune_with_ray(baseline_weights=None, num_samples=10):
    if not ray.is_initialized():
        ray.init()

    search_space = {
        "lr0": tune.loguniform(1e-5, 1e-2),
        "epochs": tune.randint(20, 120),
        "batch": tune.choice([8, 16, 32]),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "dropout": tune.uniform(0.0, 0.3),
        "imgsz": tune.choice([320, 640])
    }

    scheduler = ASHAScheduler(metric="map50_95", mode="max", max_t=150, grace_period=5, reduction_factor=2)

    analysis = tune.run(
        tune.with_parameters(train_yolo_ray, baseline_weights=baseline_weights, data="src/dataset/yolo/dataset.yml", project_root="src/runs/detect"),
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1 if settings.YOLO_DEVICE.startswith("cuda") else 0}
    )

    best_config = analysis.get_best_config(metric="map50_95", mode="max")
    ray.shutdown()

    class BestHP:
        def __init__(self, values):
            self.values = values

    return BestHP(best_config)
