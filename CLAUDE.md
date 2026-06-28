# Emotion-Lens

## Stack
- **ML:** TensorFlow 2.12 (FER2013 model, 48×48 grayscale, 7 emotions)
- **UI:** Streamlit (7 pages)
- **API:** FastAPI (inference endpoints)
- **Visualization:** Plotly + Matplotlib
- **Deployment:** Streamlit Cloud (main app), Railway (API)

## Dev commands
- `streamlit run streamlit_app.py` — launch dashboard
- `uvicorn api_server:app --reload --host 0.0.0.0 --port 8000` — start API
- `python train.py` — train new model
- `python inference.py` — test inference

## Key conventions
- 4-space indent for Python
- 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Config in `utils/config.py` (emotion colors, emojis, valence/arousal)
- Pages in `pages/` directory (auto-discovered by Streamlit)
- Model file: `emotion_model.h5` (not committed)
