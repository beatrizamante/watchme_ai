Installation Guide
=================

Step 1: System Setup
-------------------

Install Python
~~~~~~~~~~~~~

Download Python 3.11 from `python.org <https://www.python.org/downloads/>`_:

.. code-block:: bash

   python --version

Install Build Tools
~~~~~~~~~~~~~~~~~~

**Windows:**

Download and install `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   sudo apt update
   sudo apt install build-essential python3-dev

**macOS:**

.. code-block:: bash

   xcode-select --install

Step 2: Project Setup
--------------------

Clone Repository
~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/beatrizamante/watchme_ai.git
   cd watchme_ai

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~

**Using venv (recommended):**

.. code-block:: bash

   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate

**Using conda:**

.. code-block:: bash

   conda create -n watchme_ai python=3.11
   conda activate watchme_ai

Step 3: Install Dependencies
---------------------------

Install Python Packages
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

Verify Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import torch, ultralytics, torchreid; print('All packages installed successfully!')"

Step 4: GPU Setup (Optional)
----------------------------

For NVIDIA GPU acceleration:

1. Install `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_
2. Install `cuDNN <https://developer.nvidia.com/cudnn>`_
3. Verify GPU detection:

.. code-block:: bash

   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Step 5: Configuration
--------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~

1. Copy the example environment file:

.. code-block:: bash

   cp .env.example .env
   cp .env.osnet.example .env.osnet
   cp .env.yolo.example .env.yolo

2. Edit `.env` with your settings:

.. code-block:: bash

   # Example configuration
   YOLO_MODEL_PATH=src/infrastructure/yolo/client/best.pt
   OSNET_SAVE_DIR=src/infrastructure/osnet/client
   ENCRYPTION_KEY=your_base64_key_here

Generate Encryption Key
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import base64
   import secrets

   # Generate a 256-bit key
   key = secrets.token_bytes(32)
   key_b64 = base64.b64encode(key).decode()
   print(f"ENCRYPTION_KEY={key_b64}")

Next Steps
---------

After installation, proceed to :doc:`quickstart` to run your first inference.
