"""Pydantic models, constants, and API key authentication."""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Constants
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MODEL_PATH = "emotion_model.h5"
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))

# CORS — comma-separated list of allowed origins (CORS_ORIGINS env var)
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:8000,http://127.0.0.1:8000,"
    "http://localhost:8501,http://127.0.0.1:8501"
)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if o.strip()
]

# API Key Authentication
API_KEY = os.environ.get("EMOTION_API_KEY", "")
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> bool:
    """Verify API key if authentication is configured."""
    if not API_KEY:
        return True
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing API key", "message": "Provide API key via Authorization: Bearer <key> header."},
        )
    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(status_code=403, detail={"error": "Invalid API key"})
    return True


# Pydantic Models


class PredictRequest(BaseModel):
    """Request body for base64 image prediction."""

    image: str
    detect_faces: bool = True


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
