"""
EmotionLens 🎭 — FastAPI REST API Server
==========================================
Exposes the emotion detection model as a REST endpoint.
Accepts base64-encoded images or direct file uploads and returns emotion predictions as JSON.

Usage:
    # Run the server (from project root)
    python api_server.py
    
    # Or with uvicorn directly
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health          — Health check
    POST /predict         — Predict emotion from a base64 image
    POST /predict-file    — Predict emotion from an uploaded image file
    GET  /                — Root info page
"""

import os
import sys
import base64
import io
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# ─── FastAPI imports ──────────────────────────────────────────
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("❌ FastAPI is not installed. Install with: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# ─── TensorFlow / model imports ──────────────────────────────
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("❌ TensorFlow is not installed. Install with: pip install tensorflow")
    sys.exit(1)

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emotion-api")

# ─── Constants ────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
MODEL_PATH = 'emotion_model.h5'
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))

# ─── Pydantic Models ──────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for base64 image prediction."""
    image: str
    """Base64-encoded image string (with or without data URI prefix)."""
    detect_faces: bool = True
    """Whether to auto-detect faces. If False, uses the full image."""


class EmotionResult(BaseModel):
    """Single face prediction result."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    bbox: Optional[List[int]] = None


class PredictResponse(BaseModel):
    """Response from a prediction request."""
    success: bool
    faces_detected: int
    results: List[EmotionResult]
    summary: Optional[str] = None
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str
    emotions: List[str]


# ─── App Initialization ──────────────────────────────────────

app = FastAPI(
    title="EmotionLens 🎭 API",
    description="Real-time facial emotion detection API powered by CNN (FER2013). "
                "Accepts base64 images or file uploads and returns emotion predictions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for easy integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading (lazy, on first request) ──────────────────

_model = None
_face_cascade = None


def get_model():
    """Lazy-load the Keras model. Returns (model, cascade)."""
    global _model, _face_cascade
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return None, None
        logger.info(f"Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            logger.error("Failed to load face cascade.")
            return _model, None
    
    return _model, _face_cascade


def preprocess_face(face_roi):
    """Preprocess a face ROI for model prediction (48×48 grayscale)."""
    roi_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    roi_array = roi_resized.astype('float32') / 255.0
    roi_array = np.expand_dims(roi_array, axis=-1)
    roi_array = np.expand_dims(roi_array, axis=0)
    return roi_array


def predict_face(model, face_roi):
    """Predict emotion on a single face ROI."""
    processed = preprocess_face(face_roi)
    predictions = model.predict(processed, verbose=0)[0]
    max_idx = int(np.argmax(predictions))
    emotion = EMOTIONS[max_idx]
    confidence = float(predictions[max_idx])
    probs = {EMOTIONS[i]: float(predictions[i]) for i in range(7)}
    return emotion, confidence, probs


def decode_base64_image(image_b64: str) -> np.ndarray:
    """
    Decode a base64 image string to a BGR numpy array.
    Handles both raw base64 and data URI formats.
    """
    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
    
    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        # Convert PIL to BGR for OpenCV
        img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def process_image(model, cascade, img_bgr, detect_faces=True):
    """Process an image and return face-level predictions."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    
    if detect_faces:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            # Fallback: use full image
            emotion, conf, probs = predict_face(model, gray)
            results.append(EmotionResult(
                emotion=emotion, confidence=conf, probabilities=probs
            ))
            return results, 1  # 1 result from full-image fallback
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            try:
                emotion, conf, probs = predict_face(model, face_roi)
                results.append(EmotionResult(
                    emotion=emotion, confidence=conf, probabilities=probs,
                    bbox=[int(x), int(y), int(w), int(h)],
                ))
            except Exception as e:
                logger.warning(f"Error predicting face at ({x},{y}): {e}")
    else:
        emotion, conf, probs = predict_face(model, gray)
        results.append(EmotionResult(
            emotion=emotion, confidence=conf, probabilities=probs
        ))
    
    return results, len(results)


def generate_summary(results: List[EmotionResult]) -> str:
    """Generate a human-readable summary of the results."""
    if not results:
        return "No faces detected."
    if len(results) == 1:
        r = results[0]
        return f"Detected: {r.emotion} ({r.confidence*100:.1f}%)"
    
    emotion_counts = {}
    for r in results:
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1
    
    total = len(results)
    parts = []
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        parts.append(f"{emotion} {pct:.0f}%")
    
    return f"Group: {', '.join(parts)}"


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API root — provides basic info and links."""
    return {
        "service": "EmotionLens 🎭 API",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict": "Predict emotion from base64 image",
            "POST /predict-file": "Predict emotion from uploaded file",
            "GET  /docs": "Swagger UI documentation",
            "GET  /redoc": "ReDoc documentation",
        },
        "emotions": EMOTIONS,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint. Confirms the server and model are operational."""
    model, cascade = get_model()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=os.path.abspath(MODEL_PATH) if os.path.exists(MODEL_PATH) else "NOT FOUND",
        emotions=EMOTIONS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_base64(request: PredictRequest = Body(...)):
    """
    Predict emotions from a base64-encoded image.
    
    Accepts a JSON body with:
    - `image`: Base64-encoded image string (with or without `data:image/...` prefix)
    - `detect_faces` (optional, default=true): Whether to auto-detect faces
    
    Returns a list of face-level predictions with emotion, confidence, and probabilities.
    """
    import time
    start = time.time()
    
    model, cascade = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    if cascade is None:
        raise HTTPException(status_code=503, detail="Face cascade not loaded.")
    
    if not request.image:
        raise HTTPException(status_code=400, detail="No image provided.")
    
    img_bgr = decode_base64_image(request.image)
    results, faces_count = process_image(model, cascade, img_bgr, request.detect_faces)
    
    elapsed_ms = round((time.time() - start) * 1000, 2)
    
    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=elapsed_ms,
    )


@app.post("/predict-file", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_file(
    file: UploadFile = File(...),
    detect_faces: bool = Form(True, description="Whether to auto-detect faces. If False, uses the full image."),
):
    """
    Predict emotions from an uploaded image file.
    
    Supports: JPG, JPEG, PNG, WEBP using multipart/form-data.
    
    Returns the same response format as /predict.
    """
    import time
    start = time.time()
    
    model, cascade = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if cascade is None:
        raise HTTPException(status_code=503, detail="Face cascade not loaded.")
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: JPG, PNG, WEBP",
        )
    
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    results, faces_count = process_image(model, cascade, img_bgr, detect_faces)
    
    elapsed_ms = round((time.time() - start) * 1000, 2)
    
    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=elapsed_ms,
    )


# ─── CLI Entry Point ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              EmotionLens 🎭  API Server                  ║
╠══════════════════════════════════════════════════════════╣
║  Endpoints:                                              ║
║    • Health:   http://{HOST}:{PORT}/health                    ║
║    • Predict:  POST http://{HOST}:{PORT}/predict              ║
║    • Upload:   POST http://{HOST}:{PORT}/predict-file         ║
║    • Docs:     http://{HOST}:{PORT}/docs                      ║
╠══════════════════════════════════════════════════════════╣
║  Model: {MODEL_PATH:<46}║
║  Emotions: {', '.join(EMOTIONS)}  ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=HOST, port=PORT)
