# Emotion-Lens — Project Overview

## 1. Project Title
**Emotion-Lens** — A real-time facial emotion recognition system using deep learning, deployed via Streamlit web app and FastAPI inference server.

## 2. Executive Summary
Emotion-Lens is a computer vision application that detects seven facial emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from images, video streams, and webcam feeds. It uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset with optional data augmentation. The project provides three interfaces: a Streamlit web app (interactive UI with live camera, image upload, analytics, training, and gamified emotion challenges), a FastAPI REST API for programmatic inference, and a local webcam inference script. It is deployable on Docker and Streamlit Cloud.

## 3. Problem Statement
Developers and researchers need an accessible, real-time emotion recognition system that can be used both interactively (via web UI) and programmatically (via API). Existing solutions are often locked behind proprietary APIs or require complex setup. This project provides an open, self-hostable alternative with training, inference, and analysis capabilities.

## 4. Objectives
- Detect 7 facial emotions from images with reasonable accuracy
- Provide real-time webcam-based emotion detection in the browser
- Expose a REST API for programmatic emotion classification
- Allow users to train custom models through the web interface
- Offer analytics dashboards for emotion trends and distribution
- Include a gamified emotion challenge feature

## 5. Key Features
- **Image emotion analysis:** Upload an image → detect face → classify emotion
- **Live webcam detection:** Real-time emotion classification from camera feed
- **Analytics dashboard:** Emotion distribution charts, confidence scores, trends
- **Model training UI:** Configure and train CNN models from the web interface
- **Model inspector:** View model architecture, weights, and performance metrics
- **Emotion game:** Interactive challenge where users pose expressions and earn points
- **REST API endpoints:** Programmatic inference via FastAPI (POST /predict, POST /predict-file, GET /health)
- **Docker deployment:** Containerized for easy deployment

## 6. System Architecture
```
User Interface Options
    ├── Streamlit App (streamlit_app.py + pages/)
    │     ├── Live Camera (page1)
    │     ├── Image Analysis (page2)
    │     ├── Analytics (page3)
    │     ├── Train Model (page4)
    │     ├── Model Inspector (page5)
    │     └── Emotion Game (page6)
    │
    ├── FastAPI Server (api_server.py)
    │     ├── GET  /             → Root info
    │     ├── GET  /health       → Health check
    │     ├── POST /predict      → Predict from base64 image
    │     └── POST /predict-file → Predict from uploaded file
    │
    └── Webcam Script (webcam_inference.py)
          └── Real-time command-line webcam inference
                │
                ▼
          CNN Model (inference.py / train.py)
                │
                ▼
          FER2013 dataset → Trained model weights
```

## 7. Tech Stack
| Category | Technology |
|---|---|
| **Language** | Python 3.x |
| **Web UI** | Streamlit |
| **API Framework** | FastAPI with Pydantic models |
| **Deep Learning** | TensorFlow / Keras (CNN) |
| **Computer Vision** | OpenCV (cv2) with Haar Cascade face detection |
| **Image Processing** | Pillow (PIL), NumPy |
| **Data Handling** | pandas, numpy |
| **Model Serialization** | Keras .h5 files (emotion_model.h5) |
| **Visualization** | Matplotlib, plotly |
| **Deployment** | Streamlit Cloud, Docker |
| **Deployment Config** | Dockerfile, packages.txt, runtime.txt |
| **API Server** | uvicorn |

## 8. Architecture Diagram
See Section 6 — the system consists of three interface options sharing a common CNN model and inference engine.

## 9. Folder Structure
```
Emotion-Lens/
├── streamlit_app.py          # Main Streamlit application entry point
├── api_server.py             # FastAPI REST API for inference
├── inference.py              # Core inference engine (model loading + prediction)
├── train.py                  # Model training script (CNN on FER2013)
├── webcam_inference.py       # Real-time webcam emotion detection script
├── pages/
│   ├── page1_live_camera.py    # Live webcam emotion detection UI
│   ├── page2_image_analysis.py # Image upload emotion analysis UI
│   ├── page3_analytics.py      # Emotion analytics dashboard
│   ├── page4_train_model.py    # In-app model training UI
│   ├── page5_model_inspector.py# Model architecture/performance viewer
│   └── page6_emotion_game.py   # Gamified emotion challenge
├── utils/
│   └── model_utils.py         # Model loading and utility functions
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python runtime version
├── packages.txt              # System packages for Streamlit Cloud
├── Dockerfile                # Docker deployment configuration
├── README.md                 # Project documentation
└── TEST.ipynb                # Jupyter notebook for testing
```

## 10. Module Overview
- **inference.py:** Loads trained Keras model, preprocesses images (grayscale, resize to 48x48), performs prediction, returns emotion label with confidence scores
- **train.py:** CNN training pipeline — downloads FER2013 dataset via kagglehub, builds model architecture (3 Conv2D blocks + MaxPooling + Dropout + Dense layers), trains with ImageDataGenerator, saves weights to .h5 file
- **api_server.py:** FastAPI server with CORS support, 4 endpoints (root info, health check, base64 prediction, file upload prediction). Uses Pydantic models for request/response validation. Lazy-loads model on first request.
- **Webcam scripts:** Real-time face detection using OpenCV Haar cascades + emotion classification frame-by-frame

## 11. Database Overview
Not applicable — this project does not use a database. No ORM, no SQL, no storage layer. Model weights are stored as `emotion_model.h5` on disk. No user data or prediction history is persisted.

## 12. API Overview
### FastAPI Endpoints
- `GET /` — API root with service info, version, and endpoint listing
- `GET /health` — Health check returning model status, path, and available emotions
- `POST /predict` — Accepts JSON body with base64-encoded image, returns emotion predictions with confidence scores and per-face bounding boxes
- `POST /predict-file` — Accepts multipart file upload (JPG/PNG/WEBP), returns same response format as /predict

## 13. Authentication & Authorization
Not applicable — no authentication or authorization is implemented. The Streamlit app and FastAPI server are open to all connections. CORS is configured to allow all origins (`allow_origins=["*"]`).

## 14. Data Flow
1. **Image input:** User provides image via webcam, file upload, or API call
2. **Face detection:** OpenCV Haar Cascade classifier (`haarcascade_frontalface_default.xml`) detects face region(s) in image. Falls back to full-image classification if no face found.
3. **Preprocessing:** Face crop is resized to 48×48 grayscale, normalized to [0,1]
4. **Inference:** Trained CNN model predicts emotion probabilities across 7 classes
5. **Output:** Emotion label with highest confidence is returned, along with full probability distribution and optional bounding box coordinates

## 15. Request Lifecycle
For the FastAPI API:
1. HTTP request received → FastAPI router → Pydantic validation
2. Lazy-load model and face cascade if not already loaded
3. Decode image (base64 or file)
4. Convert to BGR (OpenCV format), then grayscale
5. Detect faces using Haar Cascade
6. Predict emotion for each face ROI
7. Return JSON response with predictions, confidence, probabilities, bounding boxes, and processing time

## 16. External Integrations
- **OpenCV Haar Cascades:** Built-in face detection (no external API call)
- **KaggleHub:** Downloads FER2013 dataset during training
- No third-party API services are integrated — all processing is local

## 17. Environment Variables
| Variable | Purpose | Default |
|---|---|---|
| `API_HOST` | FastAPI server bind address | `0.0.0.0` |
| `API_PORT` | FastAPI server port | `8000` |

Note: These are only used by the API server (api_server.py). The Streamlit app does not require any environment variables.

## 18. Configuration
Configuration is minimal and mostly hardcoded:
- Model file path: `emotion_model.h5` (hardcoded in inference.py and api_server.py)
- Image size: 48×48 grayscale
- Emotion labels: ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] (7 classes)
- CNN architecture: 3 Conv2D blocks (32→64→128 filters) + Flatten + Dense(1024) + Dropout(0.5) + Softmax(7)
- No external configuration files

## 19. Security Measures
- No authentication, no authorization, no rate limiting
- No input sanitization beyond basic image validation in OpenCV
- No HTTPS enforcement
- CORS allows all origins
- This is a local/development-grade application with no production security measures

## 20. Logging & Monitoring
Basic Python logging configured in api_server.py using the logging module. Streamlit and training scripts use print() statements for progress/debugging. No structured logging, no monitoring, no metrics collection.

## 21. Error Handling
Minimal try/except blocks around file loading and model inference. FastAPI uses HTTPException for structured error responses (400 for bad input, 503 for model not loaded). Training scripts exit on failure.

## 22. Performance Optimizations
- Model is loaded once and cached in memory for repeated inference (lazy loading)
- Webcam inference uses frame skipping for real-time performance
- No batch processing, no GPU-specific optimizations

## 23. Deployment Architecture
- **Streamlit Cloud:** Configured via `packages.txt` (system deps) and `runtime.txt` (Python version)
- **Docker:** Containerized with `Dockerfile` for any Docker-compatible hosting
- **FastAPI standalone:** Run with `uvicorn api_server:app --host 0.0.0.0 --port 8000`

## 24. Testing Strategy
No formal test framework found. The only testing artifact is `TEST.ipynb` — a Jupyter notebook for manual experimentation/validation. No pytest, unittest, or CI pipeline.

## 25. Development Workflow
No CONTRIBUTING.md found. No documented conventions.

## 26. Known Limitations
- **Moderate accuracy:** CNN trained on FER2013 (~65-70% accuracy ceiling for this dataset)
- **Frontal face only:** Haar cascade fails on profile/angled faces
- **No GPU support:** Inference runs on CPU (slow for real-time HD video)
- **No data persistence:** No storage for prediction history or analytics
- **Security:** Zero authentication; not suitable for production/public deployment without hardening
- **Model retraining:** The training UI is basic; advanced users would use the CLI `train.py` directly

## 27. Future Roadmap
No documented roadmap found. No TODO comments or open issues available.

## 28. Troubleshooting
- **Model not loading:** Ensure `emotion_model.h5` file exists in the project root. Run `python train.py` to train a new model if missing.
- **Webcam not working:** Verify camera permissions. OpenCV may need additional system packages (`packages.txt` lists common ones).
- **API not responding:** Run `pip install -r requirements.txt` and verify uvicorn is installed. Check `API_HOST` and `API_PORT` env vars if customized.

## 29. FAQ
- **How to run the Streamlit app?** `streamlit run streamlit_app.py`
- **How to run the API?** `uvicorn api_server:app --reload` (optionally set `API_HOST` and `API_PORT`)
- **How to train a new model?** `python train.py` or use the Train Model page in the Streamlit app.
- **What emotions are detected?** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- **API docs?** Open `/docs` or `/redoc` when the API server is running for Swagger UI.

## 30. Contributing Guidelines
Not yet defined. No CONTRIBUTING.md file exists in the repository.

## 31. License
No license file found in the repository root.

## 32. Maintainers & Contacts
No author/maintainer information specified in source files.
