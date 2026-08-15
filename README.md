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
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square" alt="TensorFlow" /></a>
</p>

---

<p align="center">
  <strong>Detect emotions from faces in real-time.</strong>
  <br />
  Live webcam, image upload, analytics dashboard, model training UI, and gamified challenges.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📹 **Live Webcam Detection** | Real-time emotion recognition with face bounding boxes |
| 🖼️ **Image Analysis** | Drag-and-drop upload with batch processing |
| 📊 **Analytics Dashboard** | Emotion distribution, trends, and heatmaps |
| 🎓 **Model Training UI** | Train CNN models in-app with configurable hyperparameters |
| 🔍 **Model Inspector** | Layer-by-layer architecture, feature maps, Grad-CAM |
| 🎮 **Emotion Challenge** | Gamified emotion recognition game with badges |
| 🔌 **REST API** | FastAPI endpoints for programmatic inference |

---

## 🎯 Detected Emotions

| Emotion | Emoji | Description |
|---------|-------|-------------|
| Angry | 😠 | Anger, frustration |
| Disgust | 🤢 | Disgust, revulsion |
| Fear | 😨 | Fear, anxiety |
| Happy | 😄 | Joy, happiness |
| Neutral | 😐 | Neutral expression |
| Sad | 😢 | Sadness, sorrow |
| Surprise | 😲 | Surprise, amazement |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam (for live features)

### Installation

```bash
# Clone the repository
git clone https://github.com/themanoj-025/Emotion-Lens.git
cd Emotion-Lens

# Install dependencies
pip install -r requirements.txt

# Train or download a model
python train.py
```

### Run the App

```bash
# Streamlit dashboard (recommended)
streamlit run streamlit_app.py

# API server
python api_server.py

# Webcam inference (CLI)
python webcam_inference.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Streamlit   │  │   FastAPI    │  │   Webcam     │          │
│  │  Dashboard   │  │   REST API   │  │   Script     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Inference Engine (inference.py)              │   │
│  │  • Model loading & caching                               │   │
│  │  • Face detection (Haar Cascade)                          │   │
│  │  • Preprocessing & prediction                             │   │
│  │  • Grad-CAM visualization                                 │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              CNN Model (emotion_model.h5)                 │   │
│  │  3 conv blocks (32→64→128) + Dense + Softmax             │   │
│  │  ~1.2M parameters | ~62% validation accuracy             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Streamlit App Pages

| Page | Description |
|------|-------------|
| 📹 **Live Camera** | Real-time webcam emotion detection |
| 🖼️ **Image Analysis** | Upload images for emotion analysis |
| 📊 **Analytics** | Emotion distribution and trends |
| 🎓 **Train Model** | Train CNN with custom settings |
| 🔍 **Model Inspector** | View model architecture and features |
| 🎮 **Emotion Game** | Gamified emotion recognition |
| ℹ️ **About** | Project information |

---

## 📁 Project Structure

```
Emotion-Lens/
├── streamlit_app.py              # Main dashboard
├── api_server.py                 # FastAPI REST API
├── inference.py                  # Core inference engine
├── train.py                      # Model training
├── webcam_inference.py           # Webcam detection
├── pages/                        # Streamlit pages
│   ├── page1_live_camera.py
│   ├── page2_image_analysis.py
│   ├── page3_analytics.py
│   ├── page4_train_model.py
│   ├── page5_model_inspector.py
│   ├── page6_emotion_game.py
│   └── page7_about.py
├── utils/
│   ├── model_utils.py            # Model loading
│   ├── emotion_utils.py          # Prediction & Grad-CAM
│   └── session_utils.py          # Session state
├── assets/
│   └── style.css                 # Custom theme
├── requirements.txt
└── Dockerfile
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict from base64 image |
| `POST` | `/predict-file` | Predict from uploaded file |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict from file
curl -X POST http://localhost:8000/predict-file \
  -F "file=@face.jpg"

# Predict from base64
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image"}'
```

---

## 🐳 Docker

```bash
# Build
docker build -t emotionlens .

# Run
docker run -p 8501:8501 -p 8000:8000 emotionlens
```

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | CNN (3 conv blocks) |
| **Input** | 48×48 grayscale images |
| **Output** | 7 emotion classes |
| **Parameters** | ~1.2M |
| **Validation Accuracy** | ~62% |
| **Dataset** | FER2013 (35,887 images) |

> 💡 **Note:** 62% accuracy is standard for FER2013 — the dataset is intentionally challenging with ambiguous expressions.

---

## 🗺️ Roadmap

- [x] Live webcam detection
- [x] Image upload analysis
- [x] Analytics dashboard
- [x] Model training UI
- [x] Model inspector
- [x] Emotion game
- [x] REST API
- [x] Docker support
- [ ] Multi-face tracking
- [ ] Video file analysis
- [ ] Emotion history
- [ ] Custom model upload

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) - Emotion dataset
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/themanoj-025">themanoj-025</a>
</p>

<p align="center">
  If you find this project useful, please give it a ⭐ star!
</p>
