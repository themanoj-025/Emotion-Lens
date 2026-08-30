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

import base64
import io
import logging
import os
import secrets
import sys

import cv2
import numpy as np
from PIL import Image

# FastAPI imports
try:
    from fastapi import APIRouter, Body, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("FastAPI is not installed. Install with: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.warning("slowapi not installed. Rate limiting disabled. Install with: pip install slowapi")
    Limiter = None

# TensorFlow / model imports
try:
    from tensorflow.keras.models import load_model
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("TensorFlow is not installed. Install with: pip install tensorflow")
    sys.exit(1)

# Structured Logging
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any as _Any

try:
    from prometheus_client import Counter, Histogram, generate_latest

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


class _StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, _Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields
        for key in ("method", "path", "status_code", "duration_ms", "faces_detected"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(log_entry, ensure_ascii=False, default=str)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("emotion-api")

# Add structured JSON file handler
try:
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _file_handler = logging.FileHandler(_log_dir / "api.log", encoding="utf-8")
    _file_handler.setFormatter(_StructuredFormatter())
    logger.addHandler(_file_handler)
except OSError:
    pass

# Constants
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MODEL_PATH = "emotion_model.h5"
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))

# API Key Authentication
# Set EMOTION_API_KEY env var to enable auth. Leave unset to disable.
API_KEY = os.environ.get("EMOTION_API_KEY", "")
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """Verify API key if authentication is configured.

    If EMOTION_API_KEY env var is set, all prediction endpoints
    require a valid Bearer token matching the configured key.
    If EMOTION_API_KEY is empty, authentication is disabled (open access).
    """
    if not API_KEY:
        return True

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing API key",
                "message": "Authentication required. Provide API key via Authorization: Bearer <key> header.",
            },
        )

    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is invalid.",
            },
        )

    return True


# Pydantic Models


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
    probabilities: dict[str, float]
    bbox: list[int] | None = None


class PredictResponse(BaseModel):
    """Response from a prediction request."""

    success: bool
    faces_detected: int
    results: list[EmotionResult]
    summary: str | None = None
    processing_time_ms: float | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_path: str
    emotions: list[str]


# ── Prometheus metrics ────────────────────────────────────────────────
if _PROM_AVAILABLE:
    EMOTION_REQUEST_COUNT = Counter(
        "emotionlens_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    EMOTION_REQUEST_LATENCY = Histogram(
        "emotionlens_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    EMOTION_PREDICTIONS = Counter(
        "emotionlens_predictions_total", "Total emotion predictions", ["emotion"])
    EMOTION_FACES_DETECTED = Counter(
        "emotionlens_faces_detected_total", "Total faces detected")

# App Initialization

app = FastAPI(
    title="EmotionLens 🎭 API",
    description="Real-time facial emotion detection API using a CNN trained on FER2013. "
    "Accepts base64 images or file uploads and returns emotion predictions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(verify_api_key)] if API_KEY else [],

# --- OpenTelemetry distributed tracing (OTEL_ENABLED=true) ---
try:
    from utils.tracing import setup_tracing
    _otel_ok = setup_tracing("emotion-lens-api")
    if _otel_ok:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
except ImportError:
    pass

    openapi_tags=[
        {
            "name": "Health",
            "description": "Service health check endpoints",
        },
        {
            "name": "Prediction",
            "description": "Emotion prediction from images (base64 or file upload)",
        },
        {
            "name": "Info",
            "description": "API information and documentation",
        },
    ],
)

# Log auth status on startup
if API_KEY:
    logger.info("✓ API key authentication enabled (EMOTION_API_KEY is set)")
else:
    logger.info("⚠ API key authentication DISABLED — set EMOTION_API_KEY env var to enable")

# CORS — restricted to localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info("✓ Rate limiting enabled (60/minute default)")
else:
    limiter = None
    logger.warning("⚠ Rate limiting DISABLED — slowapi not installed")

# Security Headers


@app.middleware("http")
async def add_security_headers(request, call_next) -> None:
    """Add security headers to every response."""
    import time as _time
    request.state.start_time = _time.time()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    )
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

    if _PROM_AVAILABLE:
        import time as _time

        path = request.url.path
        EMOTION_REQUEST_COUNT.labels(
            method=request.method, endpoint=path, status=response.status_code
        ).inc()
        if hasattr(request.state, "start_time"):
            EMOTION_REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(
                _time.time() - request.state.start_time
            )

    return response


# Model Loading (lazy, on first request)

_model = None
_face_cascade = None


def get_model() -> None:
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
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            logger.error("Failed to load face cascade.")
            return _model, None

    return _model, _face_cascade


def preprocess_face(face_roi) -> None:
    """Preprocess a face ROI for model prediction (48×48 grayscale)."""
    roi_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    roi_array = roi_resized.astype("float32") / 255.0
    roi_array = np.expand_dims(roi_array, axis=-1)
    roi_array = np.expand_dims(roi_array, axis=0)
    return roi_array


def predict_face(model, face_roi) -> None:
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
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        # Convert PIL to BGR for OpenCV
        img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return img_array
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def process_image(model, cascade, img_bgr, detect_faces=True) -> None:
    """Process an image and return face-level predictions."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    if detect_faces:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # Fallback: use full image
            emotion, conf, probs = predict_face(model, gray)
            results.append(EmotionResult(emotion=emotion, confidence=conf, probabilities=probs))
            return results, 1  # 1 result from full-image fallback

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            try:
                emotion, conf, probs = predict_face(model, face_roi)
                results.append(
                    EmotionResult(
                        emotion=emotion,
                        confidence=conf,
                        probabilities=probs,
                        bbox=[int(x), int(y), int(w), int(h)],
                    )
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Error predicting face at ({x},{y}): {e}")
    else:
        emotion, conf, probs = predict_face(model, gray)
        results.append(EmotionResult(emotion=emotion, confidence=conf, probabilities=probs))

    return results, len(results)


def generate_summary(results: list[EmotionResult]) -> str:
    """Generate a human-readable summary of the results."""
    if not results:
        return "No faces detected."
    if len(results) == 1:
        r = results[0]
        return f"Detected: {r.emotion} ({r.confidence * 100:.1f}%)"

    emotion_counts = {}
    for r in results:
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    total = len(results)
    parts = []
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        parts.append(f"{emotion} {pct:.0f}%")

    return f"Group: {', '.join(parts)}"


# ── API v1 Router ──────────────────────────────────────────────────────
v1_router = APIRouter(prefix="/api/v1")


@v1_router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_base64(request: PredictRequest = Body(...)) -> None:
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

    if _PROM_AVAILABLE:
        EMOTION_FACES_DETECTED.inc(faces_count)
        for r in results:
            EMOTION_PREDICTIONS.labels(emotion=r.emotion).inc()

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=elapsed_ms,
    )


@v1_router.post("/predict-file", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_file(
    file: UploadFile = File(...),
    detect_faces: bool = Form(
        True, description="Whether to auto-detect faces. If False, uses the full image."
    ),
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
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    results, faces_count = process_image(model, cascade, img_bgr, detect_faces)

    if _PROM_AVAILABLE:
        EMOTION_FACES_DETECTED.inc(faces_count)
        for r in results:
            EMOTION_PREDICTIONS.labels(emotion=r.emotion).inc()

    elapsed_ms = round((time.time() - start) * 1000, 2)

    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=elapsed_ms,
    )


app.include_router(v1_router)


# Root / health / metrics (unversioned — for probes and monitoring)


@app.get("/", tags=["Info"])
async def root() -> None:
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
async def health_check() -> None:
    """Health check endpoint. Confirms the server and model are operational."""
    model, _cascade = get_model()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=os.path.abspath(MODEL_PATH) if os.path.exists(MODEL_PATH) else "NOT FOUND",
        emotions=EMOTIONS,
    )


@app.get("/metrics", tags=["Info"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not _PROM_AVAILABLE:
        return {"status": "prometheus_client not installed"}
    return Response(content=generate_latest(), media_type="text/plain")


# CLI Entry Point

if __name__ == "__main__":
    import uvicorn

    print(
        f"""
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
║  Emotions: {", ".join(EMOTIONS)}  ║
╚══════════════════════════════════════════════════════════╝
    """
    )

    uvicorn.run(app, host=HOST, port=PORT)
