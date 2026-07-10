# Database Map — Emotion-Lens

## Data Storage
No database is used. Model weights are stored as files on disk.

| File | Format | Purpose |
|------|--------|---------|
| `emotion_model.h5` | Keras HDF5 | Trained CNN model weights |
| `haarcascade_frontalface_default.xml` | OpenCV XML | Face detection cascade |

## Data Entities
| Entity | Storage | Description |
|--------|---------|-------------|
| Model Weights | File (h5) | CNN weights for emotion classification |
| Session Predictions | Streamlit session_state | In-memory prediction history |
| Training History | Streamlit session_state | In-memory training progress |
