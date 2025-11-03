Configuration Guide
===================

This guide covers the configuration options for the WatchMe AI Backend, including environment variables, model settings, and deployment configurations.

Environment Variables
---------------------

The WatchMe AI Backend uses environment variables for configuration management.

Create a ``.env`` file in your project root with the following settings:

Core Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cp .env.example .env

   # Basic server configuration
   HOST=0.0.0.0
   PORT=5000
   DEBUG=false

   # Security
   ENCRYPTION_KEY=your_base64_encryption_key_here

Create a ``.env.yolo`` and a ``.env.osnet``  file in your project root with the following settings:

Model Configuration
-------------------

.. code-block:: bash

   cp .env.yolo.example .env.yolo
   cp .env.osnet.example .env.osnet

   # YOLO Configuration
   YOLO_MODEL_PATH=src/infrastructure/yolo/client/best.pt
   YOLO_BATCH_SIZE=16
   YOLO_LEARNING_RATE=0.001
   YOLO_LOSS_FUNC=AdamW
   YOLO_DROPOUT=0.0
   YOLO_DEVICE=0
   YOLO_DATASET_PATH=src/dataset/yolo/dataset.yml
   YOLO_PROJECT_PATH=src/dataset/yolo/runs/detect


   # OSNet Configuration
   OSNET_EPOCHS=250
   OSNET_LEARNING_RATE=0.0003
   OSNET_WEIGHT_DECAY=0.0005
   OSNET_BATCH_SIZE=32
   OSNET_OPTIMIZER=adam
   OSNET_NUM_CLASSES=751
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

Performance Settings
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Processing Limits
   MAX_VIDEO_SIZE_MB=500
   MAX_IMAGE_SIZE_MB=50
   FRAME_PROCESSING_LIMIT=10
   WEBSOCKET_TIMEOUT=300

Generating Encryption Key
-------------------------

The encryption key is used to secure person embeddings. Generate a secure key:

.. code-block:: python

   import base64
   import secrets

   # Generate a 256-bit key
   key = secrets.token_bytes(32)
   key_b64 = base64.b64encode(key).decode()
   print(f"ENCRYPTION_KEY={key_b64}")

Deployment Configuration
-------------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # .env.development
   DEBUG=true
   HOST=localhost
   PORT=5000
   LOG_LEVEL=DEBUG

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # .env.production
   DEBUG=false
   HOST=0.0.0.0
   PORT=443
   LOG_LEVEL=INFO
   USE_HTTPS=true
   SSL_CERT_PATH=/path/to/cert.pem
   SSL_KEY_PATH=/path/to/key.pem

Docker Configuration
~~~~~~~~~~~~~~~~~~~~

Example ``docker-compose.yml``:

.. code-block:: yaml

   version: '3.8'
   services:
     watchme-ai:
       build: .
       ports:
         - "5000:5000"
       environment:
         - HOST=0.0.0.0
         - PORT=5000
       volumes:
         - ./models:/app/models
         - ./datasets:/app/datasets
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]

Logging Configuration
---------------------

Configure logging levels and outputs:

.. code-block:: python

   import logging

   # Basic logging setup
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('watchme_ai.log'),
           logging.StreamHandler()
       ]
   )

Advanced Configuration
----------------------

Custom Model Paths
~~~~~~~~~~~~~~~~~~~

For custom trained models:

.. code-block:: bash

   # Custom YOLO model
   YOLO_MODEL_PATH=/path/to/your/custom_yolo.pt

   # Custom OSNet checkpoint
   OSNET_CHECKPOINT_PATH=/path/to/your/osnet_checkpoint.pth.tar

Troubleshooting Configuration
-----------------------------

Common Issues
~~~~~~~~~~~~~

**Invalid encryption key:**

.. code-block:: bash

   # Ensure key is base64 encoded
   python -c "import base64; print(base64.b64encode(b'your-32-byte-key-here').decode())"

**Model path errors:**

.. code-block:: bash

   # Check if model files exist
   ls -la src/infrastructure/yolo/client/best.pt
   ls -la src/infrastructure/osnet/client/

**GPU not detected:**

.. code-block:: bash

   # Verify CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"

Validation
~~~~~~~~~~

Validate your configuration:

.. code-block:: python

   from config import YOLOSettings, OSNetSettings, KeySetting

   # Test configuration loading
   yolo_config = YOLOSettings()
   osnet_config = OSNetSettings()
   key_config = KeySetting()

   print("Configuration loaded successfully!")
