"""
Page 1: 🎥 Live Camera — Real-time webcam emotion detection
Uses streamlit-webrtc for browser-native camera access with frame processing.
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils.emotion_utils import (
    apply_gradcam_overlay,
    apply_temporal_smoothing,
    compute_gradcam,
    compute_positivity_score,
    predict_emotion,
    render_mood_music_card,
)
from utils.model_utils import (
    EMOTION_CONFIG,
    EMOTIONS,
    PLOTLY_THEME,
    load_face_cascade,
    load_model_cached,
)
from utils.session_utils import add_prediction, add_snapshot


def show() -> None:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🎥 Live Emotion Camera</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Real-time facial emotion detection using a CNN
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model and cascade
    model = load_model_cached()
    face_cascade = load_face_cascade()

    if model is None:
        st.error(
            "⚠️ Model file not found. Please train a model first or place `emotion_model.h5` in the project root."
        )
        st.page_link(
            "streamlit_app.py?page=Train+Model",
            label="➡️ Go to Train Model page",
            icon="🏋️",
        )
        return

    if face_cascade is None:
        st.error("❌ Face cascade classifier failed to load. OpenCV may be misconfigured.")
        return

    # Initialize state variables
    if "temporal_buffer" not in st.session_state:
        st.session_state.temporal_buffer = []
    if "locked_prediction" not in st.session_state:
        st.session_state.locked_prediction = None
    if "lock_start_time" not in st.session_state:
        st.session_state.lock_start_time = None
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0

    # Controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.button("▶ Start Camera", type="primary", use_container_width=True)

    with col2:
        snapshot_btn = st.button(
            "📸 Snapshot",
            use_container_width=True,
            disabled=st.session_state.current_prediction is None,
            help="Capture the current frame with its emotion prediction to the gallery.",
        )
        # Snapshot logic: capture current prediction when button is clicked
        if snapshot_btn and st.session_state.current_prediction:
            pred = st.session_state.current_prediction
            # We store a placeholder image since we can't capture the exact frame post-hoc
            placeholder_img = Image.new("RGB", (100, 100), color="#1C2128")
            add_snapshot(placeholder_img, pred["emotion"], pred["confidence"])
            st.success(
                f"📸 Snapshot saved: {EMOTION_CONFIG[pred['emotion']]['emoji']} {pred['emotion']}"
            )

    with col3:
        st.button(
            "🔒 Lock Frame",
            use_container_width=True,
            disabled=st.session_state.current_prediction is None,
        )

    with col4:
        if st.session_state.locked_prediction:
            if st.button("🔓 Unlock", use_container_width=True):
                st.session_state.locked_prediction = None
                st.session_state.lock_start_time = None

    # Grad-CAM toggle (below controls, full width)
    enable_gradcam = st.checkbox(
        "🔥 Grad-CAM Live Overlay",
        value=False,
        help="When enabled, overlays a Grad-CAM heatmap on detected faces showing which facial features (eyes, mouth, brow) the CNN focuses on for its prediction. May reduce FPS.",
    )
    if enable_gradcam:
        st.info(
            "🔬 Grad-CAM active: heatmap shows where the CNN is looking — red/orange regions indicate the most influential pixels for the predicted emotion."
        )

    # Show lock info
    if st.session_state.locked_prediction:
        locked = st.session_state.locked_prediction
        st.info(
            f"🔒 Frame Locked: {EMOTION_CONFIG[locked['emotion']]['emoji']} {locked['emotion']} ({locked['confidence'] * 100:.1f}%) — press Unlock to release"
        )

    # WebRTC Implementation
    st.markdown("---")

    # Try streamlit-webrtc first, fallback to OpenCV
    use_webrtc = st.checkbox(
        "Use WebRTC (browser-native camera, recommended)",
        value=True,
        help="Uses browser's built-in camera API via WebRTC. More compatible than OpenCV.",
    )

    if use_webrtc:
        _render_webrtc_camera(model, face_cascade, enable_gradcam)
    else:
        _render_opencv_fallback(model, face_cascade, enable_gradcam)

    # Emotion Dashboard Area
    st.markdown("---")

    current_pred = st.session_state.get("current_prediction", None)

    if current_pred:
        # Apply temporal smoothing to reduce flickering
        smoothed = apply_temporal_smoothing(
            st.session_state.temporal_buffer, current_pred, window=5
        )

        # Layout: Left = Dominant Emotion Card, Right = All Emotions Bar Chart
        left_col, right_col = st.columns([1, 1.5])

        with left_col:
            _render_dominant_emotion_card(smoothed)
            render_mood_music_card(smoothed["emotion"], smoothed["confidence"])

        with right_col:
            _render_emotion_bars(smoothed)

        # Emotion History
        st.markdown("### 📈 Emotion History (last 60 frames)")
        _render_emotion_history()

        # Smile detector alert
        if smoothed["emotion"] == "Happy" and smoothed["confidence"] > 0.70:
            st.balloons()
            st.success("😊 **You're Smiling!** Keep it up!")

        # Positivity meter
        positivity = compute_positivity_score(smoothed["probabilities"])
        st.markdown("### 💚 Positivity Meter")
        _render_positivity_gauge(positivity)

    else:
        st.info("👆 Press **▶ Start Camera** above to begin real-time emotion detection.")
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; color: #8B949E;">
                <p style="font-size: 3rem;">🎭</p>
                <p>Your emotions will appear here in real-time</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Snapshot Gallery
    if st.session_state.get("snapshots"):
        st.markdown("---")
        st.markdown("### 📸 Snapshot Gallery")
        cols = st.columns(min(4, len(st.session_state.snapshots)))
        for i, snap in enumerate(st.session_state.snapshots[-8:]):
            with cols[i % 4]:
                st.image(snap["image"], use_container_width=True)
                config = EMOTION_CONFIG.get(snap["emotion"], {})
                st.markdown(
                    f"<p style='text-align: center;'>{config.get('emoji', '')} {snap['emotion']} {snap['confidence'] * 100:.0f}%</p>",
                    unsafe_allow_html=True,
                )



from pages.camera_renderers import (
    _render_webrtc_camera,
    _render_opencv_fallback,
)
