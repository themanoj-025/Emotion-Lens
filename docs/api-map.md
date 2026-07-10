# API Map — Emotion-Lens

## FastAPI Endpoints
| Method | Endpoint | Input | Output | Purpose |
|--------|----------|-------|--------|---------|
| GET | `/` | None | Service info | API root |
| GET | `/health` | None | Status, model info | Health check |
| POST | `/predict` | `{"image": "<base64>"}` | Emotion prediction | Predict from base64 |
| POST | `/predict-file` | Multipart file upload | Emotion prediction | Predict from file |

## External Integrations
| Service | Purpose | Auth |
|---------|---------|------|
| **OpenCV Haar Cascades** | Face detection (built-in) | None |
| **KaggleHub** | Download FER2013 dataset (training only) | None |

## Response Format
```json
{
  "success": true,
  "faces_detected": 1,
  "results": [{
    "emotion": "Happy",
    "confidence": 0.873,
    "probabilities": {"Angry": 0.01, ..., "Surprise": 0.02},
    "bbox": [120, 80, 180, 200]
  }],
  "processing_time_ms": 45.2
}
```
