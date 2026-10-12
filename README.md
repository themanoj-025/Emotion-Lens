<p align="center">
  <img src="https://img.shields.io/badge/EmotionLens-Facial%20Emotion%20Recognition-purple?style=for-the-badge" alt="EmotionLens Logo" />
</p>

<h1 align="center">EmotionLens</h1>

<p align="center">
  <strong>Real-time facial emotion recognition platform featuring interactive Streamlit interfaces, standalone OpenCV inference, and a FastAPI prediction service.</strong>
</p>

<p align="center">
  <a href="https://github.com/themanoj-025/Emotion-Lens/actions"><img src="https://img.shields.io/github/actions/workflow/status/themanoj-025/Emotion-Lens/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License MIT" /></a>
  <a href="https://github.com/themanoj-025/Emotion-Lens/stargazers"><img src="https://img.shields.io/github/stars/themanoj-025/Emotion-Lens?style=social" alt="GitHub Stars" /></a>
  <a href="https://github.com/themanoj-025/Emotion-Lens/issues"><img src="https://img.shields.io/github/issues/themanoj-025/Emotion-Lens?style=flat-square" alt="GitHub Issues" /></a>
  <a href="https://github.com/themanoj-025/Emotion-Lens/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome" /></a>
</p>

---

## Overview

EmotionLens analyzes facial micro-expressions in real time, classifying human emotions across primary affective categories. The repository offers a full spectrum of interfaces: a multi-page Streamlit application for interactive webcam capture and model re-training, an ultra-low-latency OpenCV script for direct camera streams, and a production-ready FastAPI REST service.

The system runs entirely locally with zero external API dependencies or cloud telemetry required.

---

## Table of Contents

- [Key Features](#key-features)
- [Recognized Emotions](#recognized-emotions)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running EmotionLens](#running-emotionlens)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Support](#support)

---

## Key Features

- **Live Webcam Classification**: Real-time face detection, bounding-box tracking, and probability scoring overlaid directly on video streams.
- **Multi-Page Streamlit Suite**: Includes dedicated views for webcam analysis, static image uploads, session aggregate metrics, in-browser model re-training, and interactive challenges.
- **FastAPI REST Service**: Production endpoints supporting base64 payloads and multipart image uploads with built-in rate limiting (`slowapi`).
- **Resilient Inference Pipeline**: Implements defensive circuit breaker safeguards and graceful error handling (`circuit_breaker.py`).
- **Offline & Private Execution**: Processes all video and image frames entirely on-device without data transmission.

---

## Recognized Emotions

The deep learning model outputs probability distributions across affective classes:

| Emotion | Expression Indicators |
| :--- | :--- |
| 😊 **Happy** | Raised cheekbones, smiling lip corners, positive valence |
| 😢 **Sad** | Lowered mouth corners, furrowed brow inner corners |
| 😡 **Angry** | Lowered brows, narrowed eyelids, tightened lips |
| 😨 **Fear** | Widened eyes, raised eyebrows, open mouth |
| 😲 **Surprise** | Dropped jaw, widened eyes, elevated eyebrows |
| 😐 **Neutral** | Relaxed facial musculature, resting state |
| 🤢 **Disgust** | Wrinkled nose, raised upper lip |

---

## Architecture

```text
                  Video / Image Input
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Webcam Video Stream          Static Image File
    (OpenCV / Streamlit)         (Upload / REST API)
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
               Haar Cascade / Face Detector
                           │
                           ▼
               Cropped & Grayscale Face
                           │
                           ▼
          Deep Learning Emotion Classifier
          (CNN / TensorFlow / Keras Model)
                           │
                           ▼
             Per-Emotion Probability Vector
             (Argmax Prediction & Confidence)
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Streamlit UI    OpenCV Window  FastAPI REST
       (Port 8501)       (Native)     (Port 8000)
```

---

## Project Structure

```text
Emotion-Lens/
├── pages/                  # Streamlit multi-page views (train, analytics, etc.)
├── utils/                  # Image preprocessing and visualization utilities
├── tests/                  # Pytest automated test suites
├── api_models.py           # Pydantic models for API request/response contracts
├── api_routes.py           # Endpoint definitions and prediction routers
├── api_server.py           # FastAPI server entry point
├── circuit_breaker.py      # Error circuit breaker pattern
├── inference.py            # Model loading and frame prediction logic
├── streamlit_app.py        # Streamlit interactive application entry point
├── train.py                # Model training and fine-tuning pipeline
├── webcam_inference.py     # Native OpenCV desktop webcam inference
├── Dockerfile              # Containerization definition
├── docker-compose.yml      # Docker Compose configuration
├── pyproject.toml          # Ruff and build configuration
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Prerequisites

- **Python**: `>= 3.10`
- **Webcam**: Required for live camera inference features
- **Operating System**: Windows, macOS, or Linux

---

## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/themanoj-025/Emotion-Lens.git
cd Emotion-Lens

# 2. Create virtual environment
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt

# 4. Create environment file
cp .env.example .env
```

---

## Running EmotionLens

### Option 1: Streamlit Interactive Dashboard

```bash
streamlit run streamlit_app.py
```
*Access the interface at http://localhost:8501.*

### Option 2: Native Desktop Webcam Inference

```bash
python webcam_inference.py
```
*Launches an OpenCV window with real-time bounding box annotations (press `q` to exit).*

### Option 3: FastAPI REST Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger docs available at http://localhost:8000/docs.*

---

## API Reference

The FastAPI server provides the following endpoints:

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health status probe |
| `POST` | `/api/v1/predict` | Predicts emotion from a JSON payload with base64-encoded image |
| `POST` | `/api/v1/predict-file` | Predicts emotion from a multipart image file upload |

### Example Request (`/api/v1/predict-file`)

```bash
curl -X POST http://localhost:8000/api/v1/predict-file \
  -F "file=@sample_face.jpg"
```

---

## Testing

```bash
# Run pytest test suite
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up --build -d

# Check running services
docker compose ps
```

---

## Roadmap

- [x] Streamlit web application with multiple interactive modes
- [x] OpenCV desktop webcam detection script
- [x] FastAPI REST server for programmatic emotion prediction
- [x] Dockerfile and compose configurations
- [ ] ONNX runtime export for accelerated edge device inference <!-- TODO: Track in GitHub issue -->
- [ ] Multi-face tracking with identity persistence across frames <!-- TODO: Track in GitHub issue -->

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Security

Please report vulnerabilities confidentially via the instructions in [SECURITY.md](SECURITY.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

- 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/themanoj-025/Emotion-Lens/issues)
- 💡 **Discussions & PRs**: [GitHub Pull Requests](https://github.com/themanoj-025/Emotion-Lens/pulls)
