# EmotionLens — Architecture

> Textual architecture of the EmotionLens real-time facial emotion recognition system (as-is; no behavior changes).

## System Overview

EmotionLens is a flat-layout TensorFlow + Streamlit application with three surfaces:

1. **Streamlit dashboard** (`streamlit_app.py` + `pages/`) — live webcam, image analysis, analytics, training UI, model inspector, emotion game.
2. **FastAPI server** (`api_server.py`) — REST endpoint for model inference (`/predict`, `/predict-file`).
3. **CLI** (`train.py`, `inference.py`, `webcam_inference.py`) — training and inference scripts.

```mermaid
graph TD
    subgraph GUI[Streamlit streamlit_app.py]
        P1[page1_live_camera]
        P2[page2_image_analysis]
        P3[page3_analytics]
        P4[page4_train_model]
        P5[page5_model_inspector]
        P6[page6_emotion_game]
        P7[page7_about]
    end

    subgraph API[FastAPI api_server.py]
        HEALTH[/health]
        PRED[/predict]
        PREDF[/predict-file]
    end

    subgraph CLI
        TRAIN[train.py]
        INF[inference.py]
        WEB[webcam_inference.py]
    end

    UTILS[utils/: model, emotion, session,
          chart, export, gradcam, smoothing, config]

    GUI --> UTILS
    API --> UTILS
    CLI --> UTILS
    UTILS --> MODEL[(CNN model / saved_model)]
```

## Data Flow (prediction path)

```
webcam frame / uploaded image
        │
        ▼
utils.emotion_utils (preprocess) ──► CNN model ──► softmax probabilities
        │
        ▼
utils.smoothing_utils (temporal smoothing for live video)
        │
        ▼
emotion label + confidence ──► Streamlit page or REST JSON
```

## Model & Training

- CNN trained via `train.py` or in-app `page4_train_model.py`; model persisted to `saved_model/` (gitignored, pulled at runtime via `ensure_model_on_cloud`).
- GradCAM (`utils/gradcam_utils.py`) provides saliency maps for explainability.

## Deployment

- Docker via root `Dockerfile` + `docker-compose.yml` (dev/prod overrides).
- `runtime.txt` (Python version) + `packages.txt` (system deps) for platform deploys.
- CI: syntax check, pyflakes (non-blocking), import/runtime checks; CodeQL + gitleaks for security.
