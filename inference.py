"""Model loading, image preprocessing, and emotion prediction."""

from __future__ import annotations

import base64
import io
import logging
import os

import cv2
import numpy as np
from PIL import Image

from api_models import EMOTIONS, MODEL_PATH, EmotionResult

logger = logging.getLogger("emotion-api")

_model = None
_face_cascade = None


def get_model():
    """Lazy-load the Keras model. Returns (model, cascade)."""
    global _model, _face_cascade

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error("Model file not found: %s", MODEL_PATH)
            return None, None
        logger.info("Loading model from %s...", MODEL_PATH)
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")

    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            logger.error("Failed to load face cascade.")
            return _model, None

    return _model, _face_cascade


def preprocess_face(face_roi) -> np.ndarray:
    """Preprocess a face ROI for model prediction (48×48 grayscale)."""
    roi_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    roi_array = roi_resized.astype("float32") / 255.0
    roi_array = np.expand_dims(roi_array, axis=-1)
    roi_array = np.expand_dims(roi_array, axis=0)
    return roi_array


def predict_face(model, face_roi) -> tuple[str, float, dict[str, float]]:
    """Predict emotion on a single face ROI."""
    processed = preprocess_face(face_roi)
    predictions = model.predict(processed, verbose=0)[0]
    max_idx = int(np.argmax(predictions))
    emotion = EMOTIONS[max_idx]
    confidence = float(predictions[max_idx])
    probs = {EMOTIONS[i]: float(predictions[i]) for i in range(7)}
    return emotion, confidence, probs


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode a base64 image string to a BGR numpy array."""
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_b64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid base64 encoding: {e}") from e

    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid image data: {e}") from e


def process_image(model, cascade, img_bgr, detect_faces=True) -> tuple[list[EmotionResult], int]:
    """Process an image and return face-level predictions."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results: list[EmotionResult] = []

    if detect_faces:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            emotion, conf, probs = predict_face(model, gray)
            results.append(EmotionResult(emotion=emotion, confidence=conf, probabilities=probs))
            return results, 1

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
                logger.warning("Error predicting face at (%d,%d): %s", x, y, e)
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

    emotion_counts: dict[str, int] = {}
    for r in results:
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    total = len(results)
    parts = []
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        parts.append(f"{emotion} {pct:.0f}%")

    return f"Group: {', '.join(parts)}"
