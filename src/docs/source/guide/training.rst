Training Guide
==============

This guide covers training custom models for both YOLO object detection and OSNet person re-identification in the WatchMe AI Backend system.

Overview
--------

The WatchMe AI Backend supports training of two main model types:

* **YOLO Models**: For person detection and bounding box extraction
* **OSNet Models**: For person re-identification and embedding generation

Each model type requires specific dataset formats and training configurations.

YOLO Training
-------------

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

**Dataset Structure:**

.. code-block:: text

   src/dataset/yolo/
   ├── images/
   │   ├── train/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── val/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── test/
   │       ├── image1.jpg
   │       ├── image2.jpg
   │       └── ...
   └── labels/
       ├── train/
       │   ├── image1.txt
       │   ├── image2.txt
       │   └── ...
       ├── val/
       │   ├── image1.txt
       │   ├── image2.txt
       │   └── ...
       └── test/
           ├── image1.txt
           ├── image2.txt
           └── ...

**Label Format (YOLO format):**

Each `.txt` file contains one line per object:

.. code-block:: text

   class_id center_x center_y width height

Where coordinates are normalized (0-1):

.. code-block:: text

   0 0.5 0.5 0.3 0.8

Example for person detection (class 0 = person):

.. code-block:: text

   0 0.456 0.392 0.123 0.654
   0 0.789 0.123 0.087 0.456

**Configuration File (`dataset.yaml`):**

.. code-block:: yaml

   # filepath: src/dataset/yolo/dataset.yaml
   path: src/dataset/yolo
   train: images/train
   val: images/val
   test: images/test

   nc: 1  # number of classes
   names: ['person']  # class names

You can use the script ``annotation_to_txt.py`` to turn a JSON annotation file into a list of .txt


.. code-block:: python

    coco_json = "src/dataset/yolo/labels/instances_val2017.json" #JSON file path
    images_dir = "src/dataset/yolo/images/val" #Images directory (can be either val/train/test)
    labels_dir = "src/dataset/yolo/labels/val" #Labels directory (can be either val/train/test)
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    PERSON_CLASS_ID = 1

    for ann in tqdm(coco["annotations"], desc="Converting"):
        if ann["category_id"] != PERSON_CLASS_ID:
            continue

        img_info = images[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]

        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w /= img_w
        h /= img_h

        txt_name = os.path.splitext(img_info["file_name"])[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)

        with open(txt_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

**Environment Variables:**

This repository comes with a trained model in src/infrastructure/yolo/client/best.pt

.. code-block:: bash

   # .env.yolo
   YOLO_MODEL_PATH=yolov11n.pt  # Base model (You can use best.pt)
   YOLO_BATCH_SIZE=16
   YOLO_LEARNING_RATE=0.001
   YOLO_LOSS_FUNC=AdamW
   YOLO_DROPOUT=0.0
   YOLO_DEVICE=0
   YOLO_DATASET_PATH=src/dataset/yolo/dataset.yml
   YOLO_PROJECT_PATH=src/dataset/yolo/runs/detect


**Training Script:**

.. code-block:: python

   # python -m src.infrastructure.yolo.core.train

    class YOLOTrainer:
        """Handle YOLO training operations"""

        def __init__(self):
            self.settings = YOLOSettings()
            self.model = None
            self.results = None

        def load_model(self, weights=None):
            """Load YOLO model"""
            self.model = yolo_client(weights) #You can just call an YOLO object
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


**Advanced Training Options:**

.. code-block:: python

   trainer.train(
       data="src/dataset/yolo/dataset.yaml",
       epochs=200,
       batch=32,
       imgsz=640,
       project="src/infrastructure/yolo/projects",
       name="advanced_person_detection",

       # Advanced parameters
       patience=50,           # Early stopping patience
       save_period=10,        # Save checkpoint every N epochs
       cache=True,           # Cache images for faster training
       device='0',           # GPU device
       workers=8,            # Number of dataloader workers
       optimizer='AdamW',    # Optimizer choice
       lr0=0.01,            # Initial learning rate
       lrf=0.01,            # Final learning rate
       momentum=0.937,       # SGD momentum
       weight_decay=0.0005,  # Optimizer weight decay
       warmup_epochs=3,      # Warmup epochs
       warmup_momentum=0.8,  # Warmup initial momentum
       box=7.5,             # Box loss gain
       cls=0.5,             # Classification loss gain
       dfl=1.5,             # Distribution focal loss gain

       # Augmentation
       hsv_h=0.015,         # Image HSV-Hue augmentation
       hsv_s=0.7,           # Image HSV-Saturation augmentation
       hsv_v=0.4,           # Image HSV-Value augmentation
       degrees=0.0,         # Image rotation (+/- deg)
       translate=0.1,       # Image translation (+/- fraction)
       scale=0.5,           # Image scale (+/- gain)
       shear=0.0,           # Image shear (+/- deg)
       perspective=0.0,     # Image perspective (+/- fraction)
       flipud=0.0,          # Image flip up-down (probability)
       fliplr=0.5,          # Image flip left-right (probability)
       mosaic=1.0,          # Image mosaic (probability)
       mixup=0.0,           # Image mixup (probability)
       copy_paste=0.0       # Segment copy-paste (probability)
   )

Resume Training
~~~~~~~~~~~~~~~

.. code-block:: python

   # Resume from checkpoint
   trainer.train(
       resume=True,  # Resume from last checkpoint
       # or specify specific checkpoint:
       # resume="src/infrastructure/yolo/projects/person_detection/weights/last.pt"
   )

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

**Using Ray Tune:**

.. code-block:: python

    # python -m src.infrastructure.yolo.core.tune
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

                self.best_params = results.get_results()
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


**Tune/Training Pipeline:**

.. code-block:: python

    # python -m src.infrastructure.yolo.pipeline
    class YOLOPipeline:
        """Complete YOLO training pipeline with resume support"""

        def __init__(self):
            self.settings = YOLOSettings()
            self.trainer = YOLOTrainer()
            self.tuner = HyperparameterTuner()
            self.baseline_weights = None
            self.best_hyperparams = None
            self.final_results = None

        def _check_for_baseline_weights(self):
            """Check if there's an existing baseline weights to resume from"""
            baseline_dir = Path(self.settings.YOLO_PROJECT_PATH) / "baseline_train"
            best_weights_path = baseline_dir / "weights" / "best.pt"

            model_path = Path(self.settings.YOLO_MODEL_PATH)

            if best_weights_path.exists():
                return True, str(best_weights_path)
            elif model_path.exists():
                return True, str(model_path)
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

            self.best_hyperparams = tune_results.get_best_result().config
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
            print(f"Final model: {self.final_results.save_dir}")
            print("="*60)

            return self.final_results


OSNet Training
--------------

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

**Dataset Structure (Market-1501 format):**

.. code-block:: text

    src/dataset/osnet/dukemtmc-vidreid/
    ├── tracklet_train/
    │   ├── 0001/
    │   │   ├── 0001_c1s1_000151_01.jpg
    │   │   └── ...
    │   └── ...
    ├── tracklet_test/
    │   ├── 0001/
    │   │   ├── 0001_c2s1_001976_01.jpg
    │   │   └── ...
    │   └── ...
    ├── query/
    │   ├── 0001/
    │   │   ├── 0001_c1s1_001051_00.jpg
    │   │   └── ...
    │   └── ...

**Filename Convention:**

.. code-block:: text

   {person_id}_{camera_id}s{sequence_id}_{frame_number}_{additional_id}.jpg

   Examples:
   0001_c1s1_000151_01.jpg  # Person 1, Camera 1, Sequence 1, Frame 151
   0042_c3s2_001234_02.jpg  # Person 42, Camera 3, Sequence 2, Frame 1234

**Data Split Configuration:**

.. code-block:: python

   # src/infrastructure/osnet/core/train.py

    data_config = {
        'root': 'src/dataset/osnet',
        'sources': ['dukemtmcvidreid'],  # Use the correct dataset name
        'targets': ['dukemtmcvidreid'],
        'height': 256,
        'width': 128,
        'combineall': False,
        'transforms': ['random_flip', 'random_crop', 'normalize'],
        'k_tfm': 1,
        'load_train_targets': False
    }

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

**Environment Variables:**

.. code-block:: bash

    # .env.osnet
    OSNET_EPOCHS=250
    OSNET_LEARNING_RATE=0.0003
    OSNET_WEIGHT_DECAY=0.0005
    OSNET_BATCH_SIZE=32
    OSNET_OPTIMIZER=adam
    OSNET_NUM_CLASSES=751        # Number of unique person IDs
    OSNET_IMG_HEIGHT=256
    OSNET_IMG_WIDTH=128
    OSNET_NUM_INSTANCES=4
    OSNET_MARGIN=0.3
    OSNET_STEPSIZE=20
    OSNET_EVAL_FREQ=30
    OSNET_PRINT_FREQ=10
    OSNET_DATASET_NAME=dukemtmcvidreid
    OSNET_ROOT_DIR=src/dataset/osnet
    OSNET_SAVE_DIR=src/infrastructure/osnet/client
    OSNET_MODEL_NAME=model.pth.tar

**Training Script:**

.. code-block:: python

   # python -m src.infrastructure.osnet.core.train

    class OSNetTrainer:
        """Handle OSNet training operations."""

        def __init__(self):
            self.settings = OSNetSettings()
            self.osnet_client = OSNetModel()
            self.datamanager = None
            self.model = None
            self._initialize_components()

        def _initialize_components(self):
            """Initialize datamanager and model."""
            self.datamanager = self.osnet_client.create_datamanager()
            num_train_pids = self.datamanager.num_train_pids
            self.model = self.osnet_client.create_osnet_model(num_classes=num_train_pids)

        def train(self, weights=None, hp=None):
            """
            Train OSNet model with optional hyperparameters.

            Args:
                weights: Path to pre-trained weights/checkpoint (optional)
                hp: Dictionary of hyperparameters (optional)

            Returns:
                results: Training results with metrics
            """
            max_epoch = (
                hp.get("max_epoch", self.settings.OSNET_EPOCHS)
                if hp
                else self.settings.OSNET_EPOCHS
            )
            lr = (
                hp.get("lr", self.settings.OSNET_LEARNING_RATE)
                if hp
                else self.settings.OSNET_LEARNING_RATE
            )
            weight_decay = (
                hp.get("weight_decay", self.settings.OSNET_WEIGHT_DECAY)
                if hp
                else self.settings.OSNET_WEIGHT_DECAY
            )
            optimizer_name = (
                hp.get("optimizer", self.settings.OSNET_OPTIMIZER)
                if hp
                else self.settings.OSNET_OPTIMIZER
            )

            optimizer = torchreid.optim.build_optimizer(
                self.model, optim=optimizer_name, lr=lr, weight_decay=weight_decay
            )

            scheduler = torchreid.optim.build_lr_scheduler(
                optimizer,
                lr_scheduler="single_step",
                stepsize=self.settings.OSNET_STEPSIZE,
            )

            start_epoch = 0
            if weights and Path(weights).exists():
                print(f"Resuming from checkpoint: {weights}")
                start_epoch = torchreid.utils.resume_from_checkpoint(
                    weights, self.model, optimizer
                )
            elif weights:
                print(f"Warning: Checkpoint not found at {weights}, starting from scratch")
            else:
                print("No checkpoint provided, starting from scratch")

            engine = torchreid.engine.VideoTripletEngine(
                self.datamanager,
                self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                label_smooth=True,
                margin=self.settings.OSNET_MARGIN,
            )

            engine.run(
                save_dir=self.settings.OSNET_SAVE_DIR,
                max_epoch=max_epoch,
                start_epoch=start_epoch,
                start_eval=max_epoch // 2,
                eval_freq=self.settings.OSNET_EVAL_FREQ,
                print_freq=self.settings.OSNET_PRINT_FREQ,
                test_only=False,
            )

            results = {
                "save_dir": self.settings.OSNET_SAVE_DIR,
                "final_epoch": max_epoch,
            }

            return results

        def get_best_model_path(self):
            """Get path to the best saved model."""
            save_dir = Path(self.settings.OSNET_SAVE_DIR)
            model_path = save_dir / self.settings.OSNET_MODEL_NAME
            return str(model_path) if model_path.exists() else None


**Advanced Training Configuration:**

.. code-block:: python

   # Advanced OSNet training with custom settings

   def advanced_training():
       # Data augmentation configuration
       transform_configs = [
           'random_flip',
           'random_crop',
           'random_erase',
           'color_jitter',
           'random_gray_scale',
           'normalize'
       ]

       # Multiple dataset sources
       datamanager = torchreid.data.VideoDataManager(
           root='src/dataset/osnet',
           sources=['market1501', 'dukemtmcreid'],  # Multiple datasets
           targets=['market1501', 'dukemtmcreid'],
           height=256,
           width=128,
           batch_size_train=64,
           batch_size_test=100,
           transforms=transform_configs,
           num_instances=4,
           num_cams=8,              # Number of cameras per identity
           num_datasets=2,          # Number of datasets
           combineall=True,         # Combine all data for training
           load_train_targets=True,
           k_tfm=1,
           train_sampler='RandomIdentitySampler'
       )

       # Model with advanced configurations
       model = torchreid.models.build_model(
           name='osnet_ibn_x1_0',   # OSNet with Instance-Batch Normalization
           num_classes=datamanager.num_train_pids,
           loss=['triplet', 'softmax'],  # Combined losses
           pretrained=True,
           use_gpu=True
       )

       # Advanced optimizer configuration
       optimizer = torchreid.optim.build_optimizer(
           model,
           optim='adamw',           # AdamW optimizer
           lr=0.00035,
           weight_decay=5e-4,
           eps=1e-8,
           amsgrad=True
       )

       # Learning rate scheduler
       scheduler = torchreid.optim.build_lr_scheduler(
           optimizer,
           lr_scheduler='cosine',   # Cosine annealing
           max_epoch=120,
           restart_epoch=40,
           eta_min=1e-7
       )

       # Training engine with advanced settings
       engine = torchreid.engine.ImageTripletEngine(
           datamanager,
           model,
           optimizer=optimizer,
           scheduler=scheduler,
           use_gpu=True,
           label_smooth=True,
           margin=0.3,              # Triplet loss margin
           weight_t=1.0,            # Triplet loss weight
           weight_x=1.0,            # Softmax loss weight
       )

       # Training execution
       engine.run(
           save_dir='src/infrastructure/osnet/client',
           max_epoch=120,
           eval_freq=5,             # Evaluate every 5 epochs
           print_freq=50,           # Print every 50 iterations
           test_only=False,
           visrank=True,            # Save ranking visualization
           visrank_topk=10,         # Top-k for visualization
           use_metric_cuhk03=False,
           ranks=[1, 5, 10, 20],    # Rank metrics to compute
           rerank=False,            # Post-processing re-ranking
           save_checkpoint=True,    # Save regular checkpoints
           resume='',               # Resume from checkpoint
       )

Model Evaluation
----------------

YOLO Evaluation
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.infrastructure.yolo.core.evaluate import YOLOEvaluator

   def evaluate_yolo():
       evaluator = YOLOEvaluator()

       # Evaluate on test set
       results = evaluator.evaluate(
           model_path="src/infrastructure/yolo/projects/person_detection/weights/best.pt",
           data_config="src/dataset/yolo/data.yaml",
           split="test"
       )

       print(f"mAP@0.5: {results['metrics/mAP50(B)']:.4f}")
       print(f"mAP@0.5:0.95: {results['metrics/mAP50-95(B)']:.4f}")
       print(f"Precision: {results['metrics/precision(B)']:.4f}")
       print(f"Recall: {results['metrics/recall(B)']:.4f}")

OSNet Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_osnet():
       # Evaluation is automatically performed during training
       # Results are saved in the save_dir

       # Manual evaluation
       model = torchreid.models.build_model(
           name='osnet_x1_0',
           num_classes=751,
           pretrained=False
       )

       # Load trained weights
       checkpoint = torch.load('src/infrastructure/osnet/client/model.pth.tar')
       model.load_state_dict(checkpoint['state_dict'])

       # Evaluate
       engine = torchreid.engine.VideoTripletEngine(
           datamanager, model, optimizer, scheduler
       )

       engine.run(
           save_dir='src/infrastructure/osnet/client',
           max_epoch=0,  # No training, just evaluation
           test_only=True,
           visrank=True,
           ranks=[1, 5, 10, 20]
       )

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

1. **Ensure high-quality annotations** for YOLO training
2. **Maintain consistent identity labeling** for OSNet
3. **Balance dataset classes** to prevent bias
4. **Use data augmentation** to improve robustness

Training Tips
~~~~~~~~~~~~~

1. **Start with pre-trained models** for faster convergence
2. **Use learning rate scheduling** for better optimization
3. **Monitor validation metrics** to prevent overfitting
4. **Save regular checkpoints** to recover from failures

Hardware Optimization
~~~~~~~~~~~~~~~~~~~~~

1. **Use mixed precision training** for faster training on modern GPUs
2. **Increase batch size** with available GPU memory
3. **Use multiple GPUs** for distributed training when available
4. **Monitor GPU utilization** to ensure efficient resource usage

This comprehensive training guide provides everything needed to train custom models for the WatchMe AI Backend system, from basic setups to advanced configurations and optimization techniques.
