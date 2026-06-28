# Emotion-Lens — Architecture

```mermaid
graph TB
    subgraph UI ["Streamlit Dashboard (7 pages)"]
        A[streamlit_app.py] --> B[page1_live_camera]
        A --> C[page2_image_analysis]
        A --> D[page3_analytics]
        A --> E[page4_train_model]
        A --> F[page5_model_inspector]
        A --> G[page6_emotion_game]
        A --> H[page7_about]
    end

    subgraph Utils ["Utility Layer"]
        I[emotion_utils.py]
        J[model_utils.py]
        K[chart_utils.py]
        L[session_utils.py]
        M[smoothing_utils.py]
        N[gradcam_utils.py]
        O[export_utils.py]
        P[config.py]
    end

    subgraph API ["FastAPI Server"]
        Q[api_server.py] --> R[/predict]
        Q --> S[/predict-file]
        Q --> T[/health]
    end

    subgraph Model ["ML Model"]
        U[emotion_model.h5<br/>(TensorFlow/Keras)]
        V[train.py]
        W[inference.py]
        X[webcam_inference.py]
    end

    UI --> Utils
    Utils --> Model
    API --> Model
    Utils --> U
```

## Key Patterns

- **7 emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise (FER2013 order)
- **Model**: TensorFlow 2.12 Keras model, 48×48 grayscale input, trained on FER2013
- **Dual interfaces**: Streamlit for interactive dashboard, FastAPI for programmatic access
- **Temporal smoothing**: `EmotionSmoother` applies sliding window across frames for stable predictions
- **Grad-CAM**: GradCAM heatmap visualization for model explainability
- **Configuration**: All emotion mappings, colors, emojis, and valence/arousal values centralized in `utils/config.py`
