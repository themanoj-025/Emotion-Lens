"""
EmotionLens 🎭 — FastAPI REST API Server
==========================================
Exposes the emotion detection model as a REST endpoint.

Usage:
    python api_server.py
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health          — Health check
    POST /predict         — Predict emotion from a base64 image
    POST /predict-file    — Predict emotion from an uploaded image file
    GET  /                — Root info page
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any as _Any

# FastAPI imports
try:
    from fastapi import Depends, FastAPI
    from fastapi.middleware.cors import CORSMiddleware
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
    Limiter = None

try:
    from prometheus_client import Counter, Histogram

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

from api_models import API_KEY, HOST, PORT, verify_api_key

# Structured Logging


class _StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, _Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("method", "path", "status_code", "duration_ms", "faces_detected"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
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

try:
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _file_handler = logging.FileHandler(_log_dir / "api.log", encoding="utf-8")
    _file_handler.setFormatter(_StructuredFormatter())
    logger.addHandler(_file_handler)
except OSError:
    pass

# ── Prometheus metrics ────────────────────────────────────────────────
if _PROM_AVAILABLE:
    EMOTION_REQUEST_COUNT = Counter(
        "emotionlens_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
    )
    EMOTION_REQUEST_LATENCY = Histogram(
        "emotionlens_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    EMOTION_PREDICTIONS = Counter("emotionlens_predictions_total", "Total emotion predictions", ["emotion"])
    EMOTION_FACES_DETECTED = Counter("emotionlens_faces_detected_total", "Total faces detected")

# ── App Initialization ────────────────────────────────────────────────

app = FastAPI(
    title="EmotionLens 🎭 API",
    description="Real-time facial emotion detection API using a CNN trained on FER2013.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(verify_api_key)] if API_KEY else [],
    openapi_tags=[
        {"name": "Health", "description": "Service health check endpoints"},
        {"name": "Prediction", "description": "Emotion prediction from images"},
        {"name": "Info", "description": "API information and documentation"},
    ],
)

# OpenTelemetry distributed tracing
try:
    from utils.tracing import setup_tracing

    _otel_ok = setup_tracing("emotion-lens-api")
    if _otel_ok:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
except ImportError:
    pass

if API_KEY:
    logger.info("✓ API key authentication enabled")
else:
    logger.info("⚠ API key authentication DISABLED")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info("✓ Rate limiting enabled (60/minute)")
else:
    limiter = None
    logger.warning("⚠ Rate limiting DISABLED")


# Security Headers + Prometheus middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers and record Prometheus metrics."""
    import time as _time

    request.state.start_time = _time.time()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

    if _PROM_AVAILABLE:
        path = request.url.path
        EMOTION_REQUEST_COUNT.labels(method=request.method, endpoint=path, status=response.status_code).inc()
        if hasattr(request.state, "start_time"):
            EMOTION_REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(
                _time.time() - request.state.start_time
            )

    return response


# Include route modules
# v1 prediction routes
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from api_models import PredictRequest, PredictResponse
from api_routes import *
from inference import decode_base64_image, generate_summary, get_model, process_image

v1_router = APIRouter(prefix="/api/v1")


@v1_router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_base64(request: PredictRequest = Body(...)):
    """Predict emotions from a base64-encoded image."""
    import time

    start = time.time()
    model, cascade = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
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

    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )


@v1_router.post("/predict-file", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_file(
    file: UploadFile = File(...),
    detect_faces: bool = Form(True, description="Whether to auto-detect faces."),
):
    """Predict emotions from an uploaded image file."""
    import time

    start = time.time()
    model, cascade = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if cascade is None:
        raise HTTPException(status_code=503, detail="Face cascade not loaded.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    import io

    import cv2
    import numpy as np
    from PIL import Image

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

    return PredictResponse(
        success=True,
        faces_detected=faces_count,
        results=results,
        summary=generate_summary(results),
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )


app.include_router(v1_router)
