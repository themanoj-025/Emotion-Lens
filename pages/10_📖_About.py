"""
Page 10: 📖 About — Documentation, model info, dataset details, tech stack, and credits.
"""
import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT
from utils.model_utils import is_model_available, load_emotion_model


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>📖 About EmotionLens 🎭</h1>
        <p>Real-time facial emotion intelligence powered by deep learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview
    **EmotionLens** is a production-grade deep learning application that detects and classifies
    facial expressions into **7 emotions**: 😠 Angry, 🤢 Disgust, 😨 Fear, 😊 Happy, 😐 Neutral,
    😢 Sad, and 😲 Surprise.

    Powered by a **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset**,
    achieving approximately **62% validation accuracy** (strong baseline for FER2013).
    """)

    # ─── Model Architecture ──
    st.markdown("## 🧠 Model Architecture")
    st.markdown("""
    | Layer | Type | Details |
    |-------|------|---------|
    | Input | 48×48 grayscale | Single channel (1) |
    | Conv Block 1 | Conv2D(32) → MaxPool → Dropout(0.25) | Feature extraction |
    | Conv Block 2 | Conv2D(64) → MaxPool → Dropout(0.25) | Mid-level features |
    | Conv Block 3 | Conv2D(128) → MaxPool → Dropout(0.25) | High-level features |
    | Flatten | — | 2D→1D |
    | Dense | 1024 units, ReLU | Fully connected |
    | Dropout | 0.5 | Regularization |
    | Output | 7 units, Softmax | Emotion probabilities |
    """)
    if is_model_available():
        model = load_emotion_model()
        if model:
            st.metric("Total Parameters", f"{model.count_params():,}")

    # ─── FER2013 Dataset ──
    st.markdown("## 📊 FER2013 Dataset")
    st.markdown("""
    - **35,887** grayscale images of faces
    - **48×48** pixels resolution
    - **7 emotion categories**
    - **28,709** training / **3,589** validation (per split)
    - Introduced at ICML 2013 Workshop on Challenges in Representation Learning
    """)

    dist = {'Angry': 3995, 'Disgust': 436, 'Fear': 4097, 'Happy': 7215,
            'Neutral': 4965, 'Sad': 4830, 'Surprise': 3171}
    fig = go.Figure(go.Bar(
        x=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in dist],
        y=list(dist.values()),
        marker_color=[EMOTION_CONFIG[e]['color'] for e in dist],
        text=[f"{v:,}" for v in dist.values()],
        textposition='outside',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Class Distribution", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ─── Performance ──
    st.markdown("## 📉 Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Validation Accuracy", "~62%", "Baseline for FER2013 CNNs")
    with col2:
        st.metric("Human Baseline", "~65%", "Estimated human performance on FER2013")
    st.markdown("""
    > 💡 62% accuracy is considered a strong baseline for FER2013 with a simple CNN.
    > State-of-the-art models achieve ~73% using ensembles and attention mechanisms.
    """)

    # ─── Limitations ──
    st.markdown("## ⚠️ Limitations")
    st.markdown("""
    - **Lighting sensitivity**: Performance degrades in low-light or overexposed conditions
    - **Head pose**: Extreme angles (>45°) reduce detection accuracy
    - **Demographic bias**: FER2013 has limited diversity; performance may vary across demographics
    - **Occlusions**: Masks, glasses, and hands covering the face can interfere
    - **Subtle expressions**: Mild or micro-expressions may be misclassified
    """)

    # ─── Tech Stack ──
    st.markdown("## 🛠️ Tech Stack")
    techs = [("Python", "3.10+"), ("TensorFlow/Keras", "Deep Learning"),
             ("OpenCV", "Computer Vision"), ("Streamlit", "Dashboard UI"),
             ("Plotly", "Interactive Charts"), ("NumPy/Pandas", "Data Processing"),
             ("WebRTC", "Real-time Video"), ("FastAPI", "REST API Server")]
    st.table(pd.DataFrame(techs, columns=["Technology", "Purpose"]))

    # ─── Changelog ──
    st.markdown("## 📋 Changelog")
    st.markdown("""
    **v2.0** (Current)
    - 10-page multi-page dashboard
    - Real-time WebRTC emotion detection with temporal smoothing
    - Batch image analysis with Grad-CAM heatmaps
    - Interactive emotion challenge game with achievements
    - Mood journal with calendar view and weekly reports
    - Group analyzer with crowd emotion summaries
    - Research mode with PCA embeddings and FPS benchmarks
    - Dark/light theme toggle
    - CSV/JSON export across all pages
    """)

    # ─── Credits ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:2rem;color:#8B949E;">
        <h3 style="color:#00D4AA;">EmotionLens 🎭</h3>
        <p>Built with ❤️ using TensorFlow, Streamlit, and OpenCV</p>
        <p style="font-size:0.9rem;">
            FER2013 dataset courtesy of ICML 2013 Workshop<br>
            Model architecture: Standard CNN for facial expression recognition
        </p>
        <p style="margin-top:1rem;font-size:0.8rem;">© 2026 EmotionLens Project</p>
    </div>
    """, unsafe_allow_html=True)
