# Architecture — Emotion-Lens

## System Architecture
```
User Interface Options
    ├── Streamlit Dashboard (streamlit_app.py + pages/)
    │     ├── 🎥 Live Camera (WebRTC + OpenCV)
    │     ├── 🖼️ Image Analysis (upload)
    │     ├── 📊 Analytics Dashboard
    │     ├── 🏋️ Train Model (GUI)
    │     ├── 🧠 Model Inspector
    │     ├── 🎯 Emotion Game
    │     └── 📖 About
    │
    ├── FastAPI REST API (api_server.py)
    │     ├── GET  /health
    │     ├── POST /predict
    │     └── POST /predict-file
    │
    └── CLI Scripts
          ├── webcam_inference.py
          └── inference.py
                │
                ▼
     CNN Model (emotion_model.h5)
     + Haar Cascade (face detection)
```

## Component Diagram
```
[Input Image] → [Face Detection] → [Preprocessing] → [CNN Model] → [Emotion Output]
   │                OpenCV             resize 48×48     .h5 file       Label + %
   │                Haar              grayscale        ~1.2M params    + Probabilities
   │                Cascade           normalize [0,1]                   + Bounding Box
   │
   └── Streamlit session_state stores prediction history
```

## Request Lifecycle (API)
1. HTTP request → FastAPI router → Pydantic validation
2. Lazy-load model + face cascade if not already loaded
3. Decode image (base64 or file)
4. Convert to grayscale, detect faces
5. Predict emotion for each face ROI
6. Return JSON: {success, faces_detected, results, summary, processing_time_ms}
