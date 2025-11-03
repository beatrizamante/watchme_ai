Requirements
=============

This page outlines the system requirements for running the complete WatchMe AI ecosystem, including the AI backend, Node.js API, and React Native frontend.

System Overview
----------------

The WatchMe AI system consists of three main components:

* **WatchMe AI Backend** (Python) - This repository
* **WatchMe Node.js API** (TypeScript) - Data management and business logic
* **WatchMe Frontend** (React Native) - Mobile application

WatchMe AI Backend Requirements
--------------------------------

Core Dependencies
~~~~~~~~~~~~~~~~~

**Python 3.11 or lower**
    The AI backend requires Python 3.11 or earlier due to compatibility with PyTorch and CUDA libraries.

    Download from: https://www.python.org/downloads/release/python-31114/

**pip (Package Manager)**
    Comes with Python installation. Ensure it's up-to-date:

    .. code-block:: bash

       python -m pip install --upgrade pip

**Virtual Environment**
    Use either ``venv`` (built-in) or ``conda`` (recommended for data science):

    Using venv:

    .. code-block:: bash

       # Create environment
       python -m venv venv

       # Activate (Windows)
       venv\Scripts\activate

       # Activate (Linux/Mac)
       source venv/bin/activate

    Using conda:

    .. code-block:: bash

       # Install conda from: https://anaconda.org/anaconda/conda
       conda create -n watchme_ai python=3.11
       conda activate watchme_ai

System Dependencies
~~~~~~~~~~~~~~~~~~~

**C++ Build Tools**
    Required for compiling certain Python packages:

    * **Windows:** Visual Studio Build Tools
        Download from: https://visualstudio.microsoft.com/downloads/?q=build+tools

    * **Linux:** build-essential package

        .. code-block:: bash

           sudo apt install build-essential

**FFmpeg (Video Processing)**
    Required for video file processing and conversion:

    Download from: https://www.ffmpeg.org/download.html

    Ensure FFmpeg is added to your system PATH.

Optional GPU Support
~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated inference (recommended for production):

**NVIDIA CUDA Toolkit**
    Download from: https://developer.nvidia.com/cuda-downloads

    Supported versions: CUDA 11.8 or 12.1

**cuDNN Library**
    Download from: https://developer.nvidia.com/cudnn

    Must match your CUDA version.

.. note::
   Ensure your PyTorch installation matches your CUDA version. Check compatibility at: https://pytorch.org/get-started/locally/

Related Components (Optional)
-----------------------------

If you're running the complete WatchMe ecosystem on the same machine:

WatchMe Node.js API
~~~~~~~~~~~~~~~~~~~

**Node.js ≥22.14.0**
    JavaScript runtime for the API backend.

    Download from: https://nodejs.org/en/download

**pnpm ≥10.14.0**
    Package manager for Node.js dependencies.

    Installation guide: https://pnpm.io/installation

**PostgreSQL (Latest)**
    Database for user and video metadata management.

    Download from: https://www.postgresql.org/download

**Docker ≥28.5.1 (Recommended)**
    For containerized database deployment.

    Installation guide: https://docs.docker.com/engine

WatchMe Frontend
~~~~~~~~~~~~~~~~

**Node.js ≥22.14.0**
    Required if Node.js is not already installed.

**npm ≥11.2.0**
    Comes with Node.js installation.

    Verify installation: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

**Expo CLI ≥54.0.21**
    For React Native development.

    .. code-block:: bash

       npm install -g expo-cli

Verification
------------

After installing the requirements, verify your setup:

.. code-block:: bash

   # Check Python version
   python --version

   # Check pip
   pip --version

   # Check CUDA (if installed)
   python -c "import torch; print(torch.cuda.is_available())"

   # Check FFmpeg
   ffmpeg -version

Next Steps
----------

Once you have the requirements installed:

1. Follow the :doc:`../usage/quickstart` guide to set up the project
2. Configure your environment using the :doc:`../usage/configuration` guide
3. Start training models with the :doc:`../guide/training` guide

Troubleshooting
---------------

**Python Version Issues**
    If you encounter compatibility issues, ensure you're using Python 3.11 or earlier. Some PyTorch/CUDA combinations don't support Python 3.12+.

**CUDA Not Detected**
    Verify your CUDA installation and ensure your PyTorch version matches your CUDA version.

**FFmpeg Not Found**
    Ensure FFmpeg is properly installed and added to your system PATH.

**Build Tool Errors**
    Install the appropriate C++ build tools for your operating system before installing Python packages.

For additional help, see our troubleshooting section in the :doc:`../usage/configuration` guide.
