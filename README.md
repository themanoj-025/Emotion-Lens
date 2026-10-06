# 😃 EmotionLens

<p align="center">
  <img src="https://img.shields.io/badge/EmotionLens-Face%20Emotion%20Detection-purple?style=for-the-badge" alt="EmotionLens Logo" />
</p>

<h1 align="center">😃 EmotionLens</h1>

<p align="center">
  <strong>Real-Time Facial Emotion Recognition System</strong>
</p>

<p align="center">
  <a href="https://github.com/themanoj-025/Emotion-Lens/actions"><img src="https://img.shields.io/github/actions/workflow/status/themanoj-025/Emotion-Lens/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="https://github.com/themanoj-025/Emotion-Lens/blob/main/LICENSE"><img src="https://img.shields.io/github/license/themanoj-025/Emotion-Lens?style=flat-square" alt="License" /></a>
  <a href="https://github.com/themanoj-025/Emotion-Lens/stargazers"><img src="https://img.shields.io/github/stars/themanoj-025/Emotion-Lens?style=social" alt="Stars" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" alt="Python" /></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square" alt="TensorFlow" /></a>
</p>

---

## 📋 Table of Contents

- [What it does](#what-it-does)
- [📸 Screenshots](#-screenshots)
- [🎯 Detected emotions](#-detected-emotions)
- [✨ Features](#-features)
- [🚀 Quick start](#-quick-start)
- [🧪 Testing](#-testing)
- [🏗️ Architecture](#️-architecture)
- [📋 Streamlit app pages](#-streamlit-app-pages)
- [📁 Project structure](#-project-structure)
- [📡 API endpoints](#-api-endpoints)
- [🐳 Docker](#-docker)
- [📊 Model details](#-model-details)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📬 Support](#-support)
- [License](#license)

---

## What it does

EmotionLens detects emotions from faces in real time using TensorFlow 2. It ships a Streamlit app with live webcam capture, image upload, an analytics dashboard, a model-training UI, and gamified challenges. The whole stack runs locally with no API keys.

## Screenshots

> To add screenshots: run `streamlit run streamlit_app.py`, capture your screen, save images to `docs/assets/`, and reference them below.
>
> **Suggested screenshots:**
> - Live webcam detection with face bounding boxes (GIF)
> - Analytics dashboard with emotion distribution
> - Model training UI with progress

---

## 🎯 Detected emotions

| Emotion | Description |
| --- | --- |
| 😊 **Happy** | Smiling, positive affect |
| 😢 **Sad** | Frowning, low affect |
| 😡 **Angry** | Anger / frustration |
| 😨 **Fear** | Fear / anxiety |
| 😲 **Surprise** | Startle / shock |
| 😐 **Neutral** | Amorphous, no strong signal |
| 😌 **Disgust** | Contempt / disgust |
| 😰 **Anxiety** | Stress / anxiety |

> [!NOTE] The model outputs per-emotion probabilities; the app presents the highest-probability label and its confidence. The exact class mapping and label order are defined in `src/model.py` (`EMOTION_LABELS`), so the README and the model never drift.

## ✨ Features

| Feature | Description |
| --- | --- |
| 🎥 **Live webcam detection** | Real-time bounding box + emotion score over a webcam feed |
| 📁 **Image upload** | Upload a single photo and get frame-by-frame results |
| 📈 **Analytics dashboard** | Per-emotion distribution and session-level summary |
| 🎮 **Gamified challenges** | Users try to "trick" the model with neutral/ambiguous faces |
| 🏋️ **Model training UI** | Adjust epochs, batch size, and architecture; retrain in the browser |
| 🐳 **Docker** | One-command run with the full stack |

## 🚀 Quick start

### Prerequisites

- Python 3.8 or newer
- TensorFlow 2.x
- Docker (optional, for the full-featured run)

### Install & run

```bash
# 1. Clone the repository
git clone https://github.com/themanoj-025/Emotion-Lens.git
cd Emotion-Lens

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download a pre-trained model (or train one in the app)
python -m src.model.download --out models

# 5. Run the Streamlit app
streamlit run streamlit_app.py
```

### Run with Docker

```bash
docker compose up --build
```

Open `http://localhost:8501` to see the dashboard.

## 🧪 Testing

```bash
# Run the test suite
pytest tests/ -v
```

## 🏗️ Architecture

```text
Emotion-Lens/
├── src/
│   ├── data/                 # Data loading + preprocessing
│   ├── model/                # TensorFlow model + download + training
│   ├── app/                  # Streamlit app + pages
│   └── utils/                # Helpers (logging, metrics)
├── models/                   # Downloaded model artifacts
├── streamlit_app.py          # Entry point
├── requirements.txt
└── README.md
```

## 📋 Streamlit app pages

| Page | Description |
| --- | --- |
| **Home / Webcam** | Live webcam capture with real-time emotion bounding boxes |
| **Upload** | Single-image emotion classification |
| **Analytics** | Per-emotion distribution + session summary |
| **Train** | In-app model training (epochs, batch, architecture) |
| **Challenges** | Gamified "trick the model" mode |

## 📁 Project structure

```
Emotion-Lens/
├── src/
│   ├── data/
│   ├── model/
│   ├── app/
│   └── utils/
├── models/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## 📡 API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/predict` | Classify an image / video frame |
| `GET` | `/api/v1/emotions` | List the supported emotion labels |
| `GET` | `/health` | Health check |

### Example usage

```bash
# Classify a locally stored image
curl -X POST http://localhost:8501/api/v1/predict \
  -F "file=@photo.jpg"
```

## 🐳 Docker

```dockerfile
# Dockerfile (build)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build + run
docker build -t emotion-lens .
docker run -p 8501:8501 emotion-lens
```

## 📊 Model details

| Item | Value |
| --- | --- |
| Framework | TensorFlow 2.x |
| Input | Face crops (aligned, fixed size) |
| Output | Per-emotion probability vector (8 emotions) |
| Inference | Deterministic post-processing of argmax + softmax |
| Training UI | In-app; architecture + hyperparameters selectable |
| Reproducibility | Seeded; record hyperparameters per run |

> [!IMPORTANT] No metric guarantees are claimed here. The model's accuracy depends on the dataset and is recorded in the training UI artifacts. Add a `MODEL_CARD.md` + CI check if this is still aspirational.

## 🗺️ Roadmap

> [!CAUTION] Checked items are built and verified. Unchecked items are tracked in the issue tracker.

- [x] Live webcam detection
- [x] Image upload
- [x] Analytics dashboard
- [x] Model training UI
- [x] Gamified challenges
- [x] Docker run
- [ ] Production-grade video stream (tracked public issue)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📬 Support

- 🐛 [Report a bug](https://github.com/themanoj-025/Emotion-Lens/issues)
- 💡 [Request a feature](https://github.com/themanoj-025/Emotion-Lens/issues)
- 📧 Email the maintainer via the issue tracker

## License

MIT License — see [LICENSE](LICENSE).
