# Emotion-Lens — Copilot Instructions

## Code conventions
- Python with 4-space indentation
- TensorFlow 2.12 for ML, Streamlit for UI, Plotly for charts
- 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## Key commands
- Streamlit UI: `streamlit run streamlit_app.py`
- API server: `uvicorn api_server:app --reload --host 0.0.0.0 --port 8000`
- Train model: `python train.py`

## Architecture
- Pages in `pages/` (auto-discovered by Streamlit)
- Utils in `utils/` (emotion_utils, model_utils, chart_utils, etc.)
- Config in `utils/config.py` (emotion mappings, colors, valence)
- Model file `emotion_model.h5` is not committed (downloads on startup)
