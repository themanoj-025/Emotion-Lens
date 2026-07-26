# MEMORY.md — Emotion-Lens

## Project Overview
**EmotionLens** is a production-grade deep learning application that detects and classifies facial expressions into **7 emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. The model uses a CNN trained on the FER2013 dataset (~62% validation accuracy).

## Business Purpose
Provide an accessible, open-source facial emotion recognition system with both interactive (Streamlit UI) and programmatic (FastAPI) interfaces.

## Tech Stack
| Category | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **Computer Vision** | OpenCV |
| **Web UI** | Streamlit |
| **API** | FastAPI |
| **Visualization** | Plotly, Matplotlib |
| **Deployment** | Docker, Streamlit Cloud |

## Architecture
```
Interfaces:
├── Streamlit App (streamlit_app.py + pages/)
├── FastAPI Server (api_server.py)
└── CLI Scripts (webcam_inference.py, inference.py)
      │
      ▼
Shared CNN Model (emotion_model.h5)
      │
      ▼
FER2013 Dataset → Trained on 35,887 grayscale 48×48 face images
```

## Features (7 Pages)
1. **Live Camera** — Real-time webcam emotion detection with bounding boxes
2. **Image Analysis** — Upload images for batch emotion classification
3. **Analytics Dashboard** — Emotion distribution charts, trends, heatmaps
4. **Train Model** — GUI-based CNN training with hyperparameter tuning
5. **Model Inspector** — Architecture viewer, feature maps, Grad-CAM
6. **Emotion Game** — Interactive challenge to recognize/pose expressions
7. **About** — Project info and documentation

## API Endpoints (FastAPI)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model status |
| POST | `/predict` | Predict emotion from base64 image |
| POST | `/predict-file` | Predict emotion from uploaded image |
| GET | `/docs` | Swagger UI documentation |

## Model Architecture (Default CNN)
| Layer | Output Shape |
|-------|-------------|
| Input (48×48 grayscale) | (48, 48, 1) |
| Conv2D(32) → MaxPool → Dropout(0.25) | (24, 24, 32) |
| Conv2D(64) → MaxPool → Dropout(0.25) | (12, 12, 64) |
| Conv2D(128) → MaxPool → Dropout(0.25) | (6, 6, 128) |
| Flatten → Dense(1024) → Dropout(0.5) | 1024 |
| Output (7 units + Softmax) | 7 |
Total parameters: ~1.2M

## Data Flow
1. Image input (webcam, file upload, or API call)
2. Face detection via OpenCV Haar Cascade
3. Preprocessing: resize to 48×48 grayscale, normalize
4. CNN predicts emotion probabilities across 7 classes
5. Output: emotion label + confidence + full probability distribution

## Environment Variables
| Variable | Default | Purpose |
|-----------|---------|---------|
| API_HOST | 0.0.0.0 | FastAPI bind address |
| API_PORT | 8000 | FastAPI port |
