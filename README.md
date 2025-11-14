# WatchMe AI

This repository contains the AI backend for the WatchMe project, providing person re-identification, object detection, and video analytics via a FastAPI server. It integrates YOLO for detection and OSNet for person re-identification, supporting both batch video processing and real-time streaming.

---

## Features

- **Person Re-identification:** Extract and compare person embeddings using OSNet.
- **Object Detection:** Detect people in images and videos using YOLO.
- **Video Processing:** Search for a person in uploaded videos or live streams.
- **API Endpoints:** REST and WebSocket endpoints for embedding creation, video search, and live tracking.
- **Encryption:** Secure person embeddings with AES encryption.

---

## Project Structure

```
watchme_ai/
├── src/
│   ├── infrastructure/
│   │   ├── yolo/            # YOLO training, inference, scripts
│   │   ├── osnet/           # OSNet training, encoding, plotting
│   ├── application/         # Use cases (embedding, prediction)
│   ├── interface/
│   │   ├── http/            # FastAPI HTTP handlers and server
│   │   ├── websocket/       # WebSocket protocol handlers
├── .env.osnet               # OSNet Training Environment variables
├── .env.yolo                # YOLO Training Environment variables
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Setup Instructions

### 1. **System Requirements**

- Python 3.10 or 3.11
- pip (latest)
- [Optional] CUDA & cuDNN for GPU acceleration
- C++ Build Tools (Visual Studio, build-essential, or Xcode CLI)
- ffmpeg (for video processing)
- Git

### 2. **Clone the Repository**

```bash
git clone https://github.com/yourusername/watchme_ai.git
cd watchme_ai
```

### 3. **Create and Activate a Virtual Environment**

**Using venv:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

**Using conda:**
```bash
conda create -n watchme_ai python=3.11
conda activate watchme_ai
```

### 4. **Install Python Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. **Configure Environment Variables**

- Copy `.env.example` to `.env` and fill in the required values (paths, keys, etc.).
- If you don't have a model, change YOLO_MODEL_PATH to yolov11n.pt only, as in YOLO_MODEL_PATH=yolov11n.pt
- Example:
  ```
  YOLO_MODEL_PATH=src/infrastructure/yolo/client/best.pt
  OSNET_SAVE_DIR=src/infrastructure/osnet/client
  ENCRYPTION_KEY=your_base64_key_here
  ```

### 6. **Prepare Datasets**

- Download and place your datasets in the specified folders (see `.env` and config).
- Example: `src/dataset/yolo/`, `src/dataset/osnet/`

---

## Running the Server

**Start the FastAPI server:**
```bash
python main.py
```
Or, using Uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

---

## API Endpoints

| Endpoint              | Method | Description                                      |
|-----------------------|--------|--------------------------------------------------|
| `/upload-embedding`   | POST   | Upload image, get encrypted person embedding      |
| `/find`     | POST   | Search for person in uploaded video           |
| `/video-stream`       | WS     | Real-time person search via WebSocket             |

---

## Development & Testing

- Use the provided scripts in `src/infrastructure/yolo/scripts/` and `src/infrastructure/osnet/scripts/` for training, evaluation, and plotting.

---

## Troubleshooting

- **ModuleNotFoundError:** Run scripts from the project root using `python -m ...`
- **CUDA errors:** Ensure CUDA and cuDNN are installed and match your PyTorch version.
- **Dataset errors:** Check that dataset paths in `.env` and config files are correct.

---

## License

MIT License

---

## Contact

For questions or support, open an issue or contact beatrizamante@hotmail.com.
