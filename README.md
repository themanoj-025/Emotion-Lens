# EmotionLens — Face Emotion Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?style=flat&logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-brightgreen?style=flat)](LICENSE)

A real-time facial emotion recognition system that detects 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from images, video streams, and webcam feeds. Uses a CNN trained on FER2013 dataset, deployed via Streamlit web app and FastAPI inference server.

---

## Overview

**EmotionLens** is a computer vision application for facial expression classification. It provides three interfaces: a Streamlit web app (interactive UI with live camera, image upload, analytics, training, and gamified challenges), a FastAPI REST API for programmatic inference, and a local webcam inference script.

### Features

- **Live webcam detection** with face bounding boxes, emotion labels, and real-time probability charts
- **Image analysis** with drag-and-drop upload, batch processing, and radar charts
- **Analytics dashboard** with emotion distribution, trends, heatmaps, and session summaries
- **Model training UI** with configurable architectures, hyperparameters, and live progress
- **Model inspector** with layer-by-layer architecture, feature maps, and Grad-CAM visualization
- **Emotion challenge game** with two modes and achievement badges
- **REST API** with endpoints for base64 and file upload prediction

### Architecture

```
User Interface Options
    ├── Streamlit App (streamlit_app.py + pages/)
    │     ├── Live Camera
    │     ├── Image Analysis
    │     ├── Analytics
    │     ├── Train Model
    │     ├── Model Inspector
    │     └── Emotion Game
    ├── FastAPI Server (api_server.py)
    │     ├── GET  /health       → Health check
    │     ├── POST /predict      → Predict from base64 image
    │     └── POST /predict-file → Predict from uploaded file
    └── Webcam Script (webcam_inference.py)
                ↓
          CNN Model (emotion_model.h5)
```

### Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Web UI** | Streamlit |
| **API Framework** | FastAPI |
| **Deep Learning** | TensorFlow / Keras (CNN) |
| **Computer Vision** | OpenCV (Haar Cascade) |
| **Visualization** | Plotly, Matplotlib |
| **Deployment** | Streamlit Cloud, Docker |

### Model Architecture

The default CNN has 3 convolutional blocks (32→64→128 filters) with max pooling, dropout, and a dense top. Total parameters: ~1.2M. Validation accuracy: ~62% (standard for FER2013 dataset).

---

## Installation

### Prerequisites

- Python 3.8+
- Webcam (for live camera features)
- Trained model file (`emotion_model.h5`)

### Setup

```bash
git clone https://github.com/themanoj-025/Emotion-Lens.git
cd Emotion-Lens
pip install -r requirements.txt
# Train or download a model
python train.py  # or download pre-trained weights
```

### Usage

```bash
# Streamlit dashboard
streamlit run streamlit_app.py

# API server (sidecar)
python api_server.py

# Webcam inference (CLI)
python webcam_inference.py
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predict from file
curl -X POST http://localhost:8000/predict-file -F "file=@face.jpg"

# Predict from base64
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "'"$(base64 -w0 face.jpg)"'"}'
```

### Dataset

Trained on FER2013 dataset (ICML 2013 Workshop) — 35,887 grayscale 48x48 face images across 7 emotion categories.

---

## Project Structure

```
├── streamlit_app.py              # Main entry point & sidebar navigation
├── api_server.py                 # FastAPI REST API server
├── inference.py                  # Core inference engine
├── train.py                      # Model training script
├── webcam_inference.py           # Real-time webcam detection
├── assets/style.css              # Custom theme styles
├── utils/
│   ├── model_utils.py            # Model loading and caching
│   ├── emotion_utils.py          # Preprocessing, prediction, Grad-CAM
│   └── session_utils.py          # Session state management
├── pages/
│   ├── page1_live_camera.py      # Real-time webcam UI
│   ├── page2_image_analysis.py   # Image upload analysis
│   ├── page3_analytics.py        # Emotion analytics dashboard
│   ├── page4_train_model.py      # In-app model training
│   ├── page5_model_inspector.py  # Model architecture viewer
│   ├── page6_emotion_game.py     # Gamified emotion challenges
│   └── page7_about.py            # Project information
└── docs/                         # Architecture documentation
```

---

## Deployment

### Streamlit Cloud

1. Upload all files including `emotion_model.h5`
2. `packages.txt` includes system deps: `libgl1-mesa-glx`, `libglib2.0-0`
3. WebRTC works with built-in STUN servers

### Docker

```bash
docker build -t emotionlens .
docker run -p 8501:8501 -p 8000:8000 emotionlens
```

---

## License

MIT
