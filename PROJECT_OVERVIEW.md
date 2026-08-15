# Emotion-Lens — Real-Time Facial Emotion Recognition

> A CNN-based facial emotion detection system detecting 7 emotions from images, video streams, and webcam feeds with Streamlit UI and FastAPI inference server.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Tech Stack & Core Technologies](#2-tech-stack--core-technologies)
- [3. High-Level Architecture](#3-high-level-architecture)
- [4. Complete Folder Structure Tree](#4-complete-folder-structure-tree)
- [5. Exhaustive File-by-File & Folder-by-Folder Breakdown](#5-exhaustive-file-by-file--folder-by-folder-breakdown)
- [6. Data Models & Schemas](#6-data-models--schemas)
- [7. API Surface](#7-api-surface)
- [8. Configuration & Environment Variables](#8-configuration--environment-variables)
- [9. Build, Run & Deployment Instructions](#9-build-run--deployment-instructions)
- [10. Data & Control Flow Walkthroughs](#10-data--control-flow-walkthroughs)
- [11. Dependency Graph Summary](#11-dependency-graph-summary)
- [12. Testing Strategy](#12-testing-strategy)
- [13. Known Issues, Technical Debt & Assumptions](#13-known-issues-technical-debt--assumptions)
- [14. Glossary](#14-glossary)
- [15. Appendix](#15-appendix)

---

## 1. Executive Summary

**Emotion-Lens** is a real-time facial emotion recognition system that classifies human facial expressions into 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. It uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset (35,887 grayscale 48x48 face images) and provides three interfaces: a multi-page Streamlit web app, a FastAPI REST API, and a local webcam inference script.

**Target users**: Developers, researchers, and hobbyists building emotion-aware applications. The tool serves as both a demo platform and a production-ready inference server.

**What problem it solves**: Emotion recognition is a foundational computer vision task with applications in user experience research, mental health monitoring, security, and human-computer interaction. Emotion-Lens provides an accessible, deployable implementation with multiple interface options.

**Why it exists**: To provide a complete, working facial emotion recognition pipeline from training to deployment, with a polished UI for non-technical users and an API for developers.

*Note: The FER2013 dataset and CNN architecture details are explicitly documented in the README and code. The target user profile is inferred from the multi-interface design.*

---

## 2. Tech Stack & Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.8+ | Primary language |
| Deep Learning | TensorFlow/Keras | 2.21.0 | CNN model for emotion classification |
| Computer Vision | OpenCV | 5.0.0 | Face detection (Haar Cascade), image preprocessing |
| Web UI | Streamlit | 1.58.0 | Interactive multi-page dashboard |
| API Framework | FastAPI | — | REST API for programmatic inference |
| Visualization | Plotly | 6.9.0 | Interactive charts (radar, bar, heatmap) |
| Visualization | Matplotlib | 3.11.1 | Static charts and Grad-CAM heatmaps |
| Data Processing | pandas | 3.0.4 | Data manipulation |
| ML Utilities | scikit-learn | 1.3.2 | Evaluation metrics, train/test split |
| Image Processing | Pillow | 10.1.0 | Image loading and manipulation |
| Dataset | KaggleHub | ≥1.0.2 | FER2013 dataset download |
| Containerization | Docker | — | Multi-stage builds (prod/dev) |
| CI/CD | GitHub Actions | — | Lint, test, security scans |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    User Interface Options                            │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Streamlit App   │  │  FastAPI Server  │  │  Webcam Script   │  │
│  │  (streamlit_app) │  │  (api_server.py) │  │  (webcam_        │  │
│  │                  │  │                  │  │   inference.py)   │  │
│  │  7 Pages:        │  │  3 Endpoints:    │  │                  │  │
│  │  • Live Camera   │  │  • /health       │  │  Real-time face  │  │
│  │  • Image Analysis│  │  • /predict      │  │  detection with  │  │
│  │  • Analytics     │  │  • /predict-file │  │  bounding boxes  │  │
│  │  • Train Model   │  │                  │  │  and labels      │  │
│  │  • Model Inspector│ │                  │  │                  │  │
│  │  • Emotion Game  │  │                  │  │                  │  │
│  │  • About         │  │                  │  │                  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                     │             │
│           └─────────────────────┼─────────────────────┘             │
│                                 │                                   │
│                                 ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Inference Engine (inference.py)            │   │
│  │                                                               │   │
│  │  • Model loading & caching (model_utils.py)                  │   │
│  │  • Image preprocessing (48x48 grayscale normalization)       │   │
│  │  • CNN prediction → 7-class probability distribution         │   │
│  │  • Grad-CAM visualization for explainability                 │   │
│  │  • Face detection via OpenCV Haar Cascade                    │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              CNN Model (emotion_model.h5)                     │   │
│  │                                                               │   │
│  │  Architecture: 3 Conv Blocks (32→64→128 filters)             │   │
│  │  + MaxPooling + Dropout + Dense Top                          │   │
│  │  Total Parameters: ~1.2M                                     │   │
│  │  Validation Accuracy: ~62% (standard for FER2013)            │   │
│  │  Training: FER2013 dataset (35,887 images, 7 classes)        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Architectural Pattern**: **Shared Inference Core** with **Multiple Interface Frontends**. The CNN model and preprocessing logic are centralized in `inference.py` and `utils/`, while three independent interfaces (Streamlit, FastAPI, webcam CLI) consume the same inference engine.

---

## 4. Complete Folder Structure Tree

```
Emotion-Lens/
├── .devcontainer/
│   └── devcontainer.json           # Dev container config
├── .dockerignore                   # Docker build exclusions
├── .editorconfig                   # Editor config
├── .env.example                    # Environment template
├── .gitattributes                  # Git attributes
├── .github/
│   ├── CODEOWNERS                  # Code ownership
│   ├── copilot-instructions.md     # AI assistant instructions
│   ├── dependabot.yml              # Dependency automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md           # Bug report template
│   │   └── feature_request.md      # Feature request template
│   ├── labeler.yml                 # Auto-labeling
│   ├── PULL_REQUEST_TEMPLATE.md    # PR template
│   └── workflows/
│       ├── ci.yml                  # CI pipeline
│       ├── codeql.yml              # Code security
│       ├── gitleaks.yml            # Secret detection
│       ├── labeler.yml             # Label automation
│       ├── maintenance.yml         # Maintenance
│       ├── stale.yml               # Stale management
│       └── welcome.yml             # New contributor welcome
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit config
├── .vscode/
│   └── settings.json               # VS Code settings
├── AGENTS.md                       # AI agent instructions
├── api_server.py                   # FastAPI REST API server
├── assets/
│   └── style.css                   # Custom Streamlit theme
├── docker-compose.dev.yml          # Docker Compose dev overrides
├── docker-compose.prod.yml         # Docker Compose prod overrides
├── docker-compose.yml              # Docker Compose base
├── Dockerfile                      # Multi-stage Docker build
├── docs/
│   ├── community/
│   │   ├── CHANGELOG.md            # Release notes
│   │   ├── CODE_OF_CONDUCT.md      # Community guidelines
│   │   ├── CONTRIBUTING.md         # Contribution guide
│   │   ├── SECURITY.md             # Security policy
│   │   └── SUPPORT.md              # Support info
│   ├── design/
│   │   ├── AppFlow.md              # Application flow
│   │   └── Design.md               # System design
│   ├── product/
│   │   └── PRD.md                  # Product requirements
│   ├── project/
│   │   ├── ImplementationPlan.md   # Implementation roadmap
│   │   ├── RiskRegister.md         # Risk assessment
│   │   ├── Rules.md                # Project rules
│   │   └── Tracker.md              # Progress tracker
│   ├── reference/
│   │   └── Glossary.md             # Domain terminology
│   └── technical/
│       ├── API.md                  # API documentation
│       ├── Deployment.md           # Deployment guide
│       ├── Schema.md               # Data schema
│       ├── SecurityAndCompliance.md # Security notes
│       ├── TechSpec.md             # Technical spec
│       └── Testing.md              # Testing docs
├── inference.py                    # Core inference engine
├── LICENSE                         # MIT License
├── Makefile                        # Convenience commands
├── packages.txt                    # System dependencies (Streamlit Cloud)
├── pages/
│   ├── page1_live_camera.py        # Real-time webcam UI
│   ├── page2_image_analysis.py     # Image upload analysis
│   ├── page3_analytics.py          # Emotion analytics dashboard
│   ├── page4_train_model.py        # In-app model training
│   ├── page5_model_inspector.py    # Model architecture viewer
│   ├── page6_emotion_game.py       # Gamified emotion challenges
│   └── page7_about.py              # Project information
├── PROJECT_ANALYSIS.md             # Repository audit
├── PROJECT_OVERVIEW.md             # This file
├── pyproject.toml                  # Python tool config
├── README.md                       # Project README
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python runtime version
├── streamlit_app.py                # Main Streamlit entry point
├── train.py                        # Model training script
├── utils/
│   ├── chart_utils.py              # Chart generation utilities
│   ├── config.py                   # Configuration management
│   ├── emotion_utils.py            # Preprocessing, prediction, Grad-CAM
│   ├── export_utils.py             # Data export utilities
│   ├── gradcam_utils.py            # Grad-CAM visualization
│   ├── model_utils.py              # Model loading and caching
│   ├── session_utils.py            # Session state management
│   └── smoothing_utils.py          # Prediction smoothing
└── webcam_inference.py             # Real-time webcam detection CLI
```

---

## 5. Exhaustive File-by-File & Folder-by-Folder Breakdown

### Root Files

#### `Emotion-Lens/streamlit_app.py`
- **File type**: Python script (entry point)
- **Purpose**: Main Streamlit application entry point. Configures page layout, initializes session state, ensures model availability, injects custom CSS, and routes to 7 page modules via sidebar navigation.
- **Key exports**: None (script entry point)
- **Key functions**:
  - `init_session_state()` — Initialize Streamlit session variables
  - `ensure_model_on_cloud()` — Auto-download model on Streamlit Cloud
  - Page routing via sidebar radio buttons
- **Pages**: Live Camera, Image Analysis, Analytics, Train Model, Model Inspector, Emotion Game, About
- **Side effects**: Reads model files, writes session state, renders Streamlit UI
- **Dependencies**: `utils.session_utils`, `utils.model_utils`, all `pages/` modules

#### `Emotion-Lens/api_server.py`
- **File type**: Python script (FastAPI server)
- **Purpose**: FastAPI REST API for programmatic emotion prediction. Provides endpoints for base64 image and file upload prediction.
- **Key endpoints**: `GET /health`, `POST /predict`, `POST /predict-file`
- **Dependencies**: `inference.py`, FastAPI, uvicorn

#### `Emotion-Lens/inference.py`
- **File type**: Python module (core engine)
- **Purpose**: Core inference engine. Handles model loading, image preprocessing (48x48 grayscale normalization), CNN prediction, and Grad-CAM visualization.
- **Key exports**: Model prediction functions, preprocessing pipeline
- **Dependencies**: TensorFlow/Keras, OpenCV, numpy

#### `Emotion-Lens/train.py`
- **File type**: Python script
- **Purpose**: Model training script. Trains the CNN on FER2013 dataset with configurable architectures and hyperparameters.
- **Dependencies**: TensorFlow/Keras, FER2013 dataset

#### `Emotion-Lens/webcam_inference.py`
- **File type**: Python script
- **Purpose**: Real-time webcam emotion detection. Uses OpenCV for face detection and the trained CNN for emotion classification.
- **Dependencies**: OpenCV, inference.py

---

### `Emotion-Lens/pages/` — Streamlit Page Modules

| File | Purpose |
|------|---------|
| `page1_live_camera.py` | Real-time webcam detection with face bounding boxes, emotion labels, and probability charts |
| `page2_image_analysis.py` | Drag-and-drop image upload, batch processing, radar charts |
| `page3_analytics.py` | Emotion distribution, trends, heatmaps, session summaries |
| `page4_train_model.py` | Configurable CNN architectures, hyperparameters, live training progress |
| `page5_model_inspector.py` | Layer-by-layer architecture, feature maps, Grad-CAM visualization |
| `page6_emotion_game.py` | Two game modes with achievement badges |
| `page7_about.py` | Project information and documentation |

---

### `Emotion-Lens/utils/` — Utility Modules

| File | Purpose |
|------|---------|
| `chart_utils.py` | Chart generation (Plotly/Matplotlib) |
| `config.py` | Configuration management |
| `emotion_utils.py` | Preprocessing, prediction, Grad-CAM |
| `export_utils.py` | Data export (CSV, JSON) |
| `gradcam_utils.py` | Grad-CAM heatmap generation |
| `model_utils.py` | Model loading, caching, availability checks |
| `session_utils.py` | Streamlit session state management |
| `smoothing_utils.py` | Prediction smoothing (temporal) |

---

### `Emotion-Lens/Dockerfile`
- **File type**: Dockerfile (multi-stage)
- **Purpose**: Multi-stage build with `prod` and `dev` targets. Based on `python:3.11-slim`, includes OpenCV runtime libs, tini for PID-1, healthcheck on Streamlit's `/_stcore/health`.
- **Build targets**: `prod` (default), `dev` (hot reload)

---

## 6. Data Models & Schemas

### Emotion Classes

```python
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = 7
```

### Model Input

- **Shape**: (1, 48, 48, 1) — single grayscale image
- **Preprocessing**: Resize to 48x48, convert to grayscale, normalize pixel values to [0, 1]

### Model Output

- **Shape**: (1, 7) — probability distribution over 7 emotions
- **Format**: Softmax probabilities summing to 1.0

### Prediction Result

```json
{
  "emotion": "Happy",
  "confidence": 0.85,
  "probabilities": {
    "Angry": 0.02,
    "Disgust": 0.01,
    "Fear": 0.03,
    "Happy": 0.85,
    "Neutral": 0.05,
    "Sad": 0.02,
    "Surprise": 0.02
  }
}
```

---

## 7. API Surface

| Method | Path | Purpose | Request | Response |
|--------|------|---------|---------|----------|
| `GET` | `/health` | Health check | — | `{"status": "healthy", "model_loaded": true}` |
| `POST` | `/predict` | Predict from base64 | `{"image": "<base64>"}` | `{"emotion": "Happy", "confidence": 0.85, "probabilities": {...}}` |
| `POST` | `/predict-file` | Predict from file upload | multipart form | `{"emotion": "Happy", "confidence": 0.85, "probabilities": {...}}` |

---

## 8. Configuration & Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `FER2013_PATH` | Path to FER2013 dataset | Auto-download via KaggleHub | No (auto-downloaded) |
| `MODEL_PATH` | Path to trained model | `emotion_model.h5` | No (train or download) |
| `STREAMLIT_SERVER_PORT` | Streamlit port | `8501` | No |
| `API_PORT` | FastAPI port | `8000` | No |

---

## 9. Build, Run & Deployment Instructions

### Prerequisites

- Python 3.8+
- Webcam (for live camera features)
- Trained model file (`emotion_model.h5`)

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/themanoj-025/Emotion-Lens.git
cd Emotion-Lens
pip install -r requirements.txt

# 2. Train or download model
python train.py

# 3. Run Streamlit dashboard
streamlit run streamlit_app.py

# 4. Run API server (separate terminal)
python api_server.py

# 5. Run webcam inference (CLI)
python webcam_inference.py
```

### Docker

```bash
docker build -t emotion-lens .
docker run -p 8501:8501 -p 8000:8000 emotion-lens
```

### Streamlit Cloud

1. Upload all files including `emotion_model.h5`
2. `packages.txt` includes system deps: `libgl1-mesa-glx`, `libglib2.0-0`

---

## 10. Data & Control Flow Walkthroughs

### Flow 1: Image Upload Analysis

1. User navigates to "Image Analysis" page
2. Uploads image via drag-and-drop or file picker
3. OpenCV detects faces using Haar Cascade
4. Each face is preprocessed: resize to 48x48, grayscale, normalize
5. CNN predicts 7-class probability distribution
6. Results displayed with radar chart, bar chart, and face annotations
7. Session statistics updated

### Flow 2: Live Webcam Detection

1. User navigates to "Live Camera" page
2. Clicks "Start Camera" button
3. Streamlit WebRTC captures webcam frames
4. Each frame: detect faces → preprocess → predict → annotate
5. Real-time display with bounding boxes, emotion labels, and probability charts
6. Session emotion distribution tracked over time

---

## 11. Dependency Graph Summary

```
streamlit_app.py
  ├── pages/page1_live_camera.py → inference.py, utils/*
  ├── pages/page2_image_analysis.py → inference.py, utils/*
  ├── pages/page3_analytics.py → utils/*
  ├── pages/page4_train_model.py → train.py, utils/*
  ├── pages/page5_model_inspector.py → inference.py, utils/*
  ├── pages/page6_emotion_game.py → inference.py, utils/*
  └── pages/page7_about.py

api_server.py → inference.py
webcam_inference.py → inference.py, OpenCV
train.py → TensorFlow/Keras, FER2013
inference.py → TensorFlow/Keras, OpenCV, utils/*
```

---

## 12. Testing Strategy

- **Framework**: Not explicitly defined (no test files in file tree)
- **Manual testing**: Via Streamlit UI and API endpoints
- **Model evaluation**: Validation accuracy ~62% on FER2013 (standard for this dataset)

---

## 13. Known Issues, Technical Debt & Assumptions

### Known Issues

1. **No automated test suite**: No test files are present in the repository.
2. **~62% validation accuracy**: This is standard for FER2013 but limits real-world applicability.
3. **Haar Cascade face detection**: Less robust than deep learning-based detectors (MTCNN, RetinaFace).

### Technical Debt

1. **Duplicate requirements**: `matplotlib` and `plotly` are listed twice in `requirements.txt`.
2. **No model versioning**: Models are stored as flat `.h5` files without version tracking.
3. **No API authentication**: The FastAPI server has no auth mechanism.

### Assumptions

1. **Model file exists**: The app assumes `emotion_model.h5` is present or can be downloaded.
2. **Webcam available**: Live camera features require a working webcam.
3. **GPU optional**: Training works on CPU but is slow; GPU significantly speeds up training.

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **FER2013** | Facial Expression Recognition 2013 dataset — 35,887 grayscale 48x48 face images across 7 emotion categories |
| **CNN** | Convolutional Neural Network — deep learning architecture for image classification |
| **Haar Cascade** | OpenCV's pre-trained face detection algorithm |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — visualization technique showing which image regions influenced the prediction |
| **WebRTC** | Web Real-Time Communication — browser technology for webcam access |
| **Streamlit** | Python framework for building interactive web apps |

---

## 15. Appendix

### FER2013 Dataset

- **Source**: ICML 2013 Workshop
- **Size**: 35,887 grayscale images
- **Resolution**: 48x48 pixels
- **Classes**: 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Download**: Via KaggleHub (`kaggle datasets download -d msambare/fer2013`)

### Model Architecture Details

```
Input (48x48x1)
  → Conv2D(32, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
  → Conv2D(64, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
  → Conv2D(128, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
  → Flatten
  → Dense(256) + ReLU + Dropout(0.5)
  → Dense(7) + Softmax
```

Total parameters: ~1.2M

---

*This document was generated as part of a comprehensive project documentation effort. Last updated: August 8, 2026.*
