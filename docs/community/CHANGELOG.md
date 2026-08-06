# Changelog

All notable changes to **Emotion-Lens** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-06-01

### Added

#### Core Model
- CNN trained on FER2013 dataset (~62% validation accuracy)
- 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 3 convolutional blocks with ~1.2M parameters
- Adam optimizer with categorical crossentropy loss
- CLI training script (`train.py`) with configurable epochs, batch size, dropout

#### Streamlit Dashboard (7 Pages)
- **Live Camera** — Real-time webcam detection with bounding boxes, emotion labels, confidence bars, temporal smoothing, Grad-CAM overlay, multi-face support
- **Image Analysis** — Drag & drop upload (up to 10 images), batch processing, radar charts, CSV/JSON export
- **Analytics Dashboard** — Emotion distribution charts, timeline, transition heatmap, positivity score, session summary
- **Train Model** — GUI-based training with dataset selection, 3 architectures, hyperparameter sliders, live progress charts
- **Model Inspector** — Layer architecture table, feature map visualizer, Grad-CAM heatmaps
- **Emotion Challenge Game** — 2 game modes (Make This Face, Guess the Emotion) with leaderboard and achievements
- **About Page** — Model details, dataset info, tech stack

#### REST API (FastAPI)
- `POST /predict` — Predict from base64 image → JSON response
- `POST /predict-file` — Predict from uploaded image file
- `GET /health` — Server health check with model status
- `GET /docs` — Auto-generated Swagger UI
- CORS enabled for cross-origin integration

#### Webcam Inference
- Standalone OpenCV webcam script (`webcam_inference.py`)
- Real-time video feed with face detection and emotion labels

#### Deployment
- Dockerfile with multi-service support (Streamlit + API)
- Streamlit Cloud configuration (`packages.txt`, `runtime.txt`)
- `API_HOST` and `API_PORT` environment variable configuration

---

## [0.1.0] — Initial Development

### Added
- Project scaffolding
- FER2013 dataset loading and preprocessing
- Basic CNN model architecture
- Initial training pipeline
