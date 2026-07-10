# Dependency Graph — Emotion-Lens

## File Dependencies
```
streamlit_app.py (entry point)
├── utils.session_utils (init_session_state)
├── utils.model_utils (is_model_available)
├── assets/style.css (custom theme)
└── pages/* (7 page modules)

pages/
├── 1_🎥_Live_Camera.py
├── 2_🖼️_Image_Analysis.py
├── 3_📊_Analytics.py
├── 4_🏋️_Train_Model.py
├── 5_🧠_Model_Inspector.py
├── 6_🎯_Emotion_Game.py
└── 7_📖_About.py

utils/
├── model_utils.py (TensorFlow model loading)
├── emotion_utils.py (preprocessing, prediction, Grad-CAM)
└── session_utils.py (session state management)

api_server.py (standalone)
├── fastapi, uvicorn
├── tensorflow.keras (model loading)
├── cv2 (OpenCV face detection)
└── Pydantic models (request/response validation)

train.py, inference.py, webcam_inference.py (standalone scripts)
```

## Critical Files
| File | Impact |
|------|--------|
| `emotion_model.h5` | Trained CNN — required for all inference |
| `inference.py` | Core inference engine shared by all interfaces |
| `streamlit_app.py` | Main web app entry point with navigation |
