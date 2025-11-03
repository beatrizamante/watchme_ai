System Requirements
==================

Hardware Requirements
--------------------

Minimum
~~~~~~~

* CPU: 4-core processor (Intel i5 or AMD equivalent)
* RAM: 8GB
* Storage: 10GB free space
* GPU: Optional (CPU inference supported)

Recommended
~~~~~~~~~~~

* CPU: 8-core processor (Intel i7 or AMD equivalent)
* RAM: 16GB or more
* Storage: 50GB free space (for datasets)
* GPU: NVIDIA GPU with 6GB+ VRAM (GTX 1060 or better)

Software Requirements
--------------------

Operating System
~~~~~~~~~~~~~~~

* Windows 10/11
* Ubuntu 18.04+ / Debian 10+
* macOS 10.15+

Python Environment
~~~~~~~~~~~~~~~~~

* Python 3.10 or 3.11 (required)
* pip 21.0+ (package manager)
* Virtual environment (recommended)

Development Tools
~~~~~~~~~~~~~~~~

* Git (version control)
* C++ Build Tools:

  * Windows: Visual Studio Build Tools
  * Linux: build-essential
  * macOS: Xcode Command Line Tools

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

* CUDA 11.8+ (for GPU acceleration)
* cuDNN 8.0+ (for GPU acceleration)
* ffmpeg (for advanced video processing)

Supported Formats
----------------

Input Formats
~~~~~~~~~~~~

* **Images:** JPEG, PNG, BMP, TIFF
* **Videos:** MP4, AVI, MOV, MKV
* **Streaming:** WebSocket, RTMP, HTTP streams

Output Formats
~~~~~~~~~~~~~

* **Embeddings:** Encrypted numpy arrays
* **Results:** JSON responses
* **Visualizations:** PNG, PDF plots
