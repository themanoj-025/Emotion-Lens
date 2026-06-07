"""
Page 1: 🎥 Live Camera — Real-time emotion detection via WebRTC.
Uses streamlit-webrtc for browser-native camera access with frame processing.
Applies temporal smoothing to stabilize predictions.
Features: multi-face support, Grad-CAM overlay, snapshot capture, emotion history.
"""
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import threading
import time
import streamlit as st
from PIL import Image
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT
from utils.model_utils import (
    load_emotion_model, load_face_cascade, detect_and_predict,
    draw_emotion_overlay, predict_emotion, preprocess_roi
)
from utils.smoothing_utils import EmotionSmoother
from utils.session_utils import init_session, record_prediction, add_snapshot
from utils.gradcam_utils import make_gradcam_heatmap, overlay_gradcam
from utils.config import positivity_score
import plotly.graph_objects as go

RTC_CONFIG = RTCConfiguration({"iceServers": [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]})


class EmotionProcessor(VideoProcessorBase):
    """Video frame processor with temporal smoothing and Grad-CAM support."""

    def __init__(self):
        self.model = load_emotion_model()
        self.face_cascade = load_face_cascade()
        self.smoother = EmotionSmoother(window=5)
        self.last_results: list[dict] = []
        self.fps_counter = 0
        self.fps = 0.0
        self._t = time.time()
        self._lock = threading.Lock()
        self.gradcam_on = False
        self.confidence_threshold = 0.3
        self.results_history: list[dict] = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # FPS calculation
        self.fps_counter += 1
        elapsed = time.time() - self._t
        if elapsed >= 1.0:
            with self._lock:
                self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self._t = time.time()

        faces = detect_and_predict(self.model, self.face_cascade, img)

        # Apply temporal smoothing to first detected face
        if faces and len(faces) > 0:
            smoothed = self.smoother.update(faces[0]['probabilities'])
            idx = int(np.argmax(smoothed))
            faces[0]['emotion'] = EMOTIONS[idx]
            faces[0]['confidence'] = smoothed[idx]
            faces[0]['probabilities'] = smoothed

        # Filter by confidence threshold
        faces = [f for f in faces if f['confidence'] >= self.confidence_threshold]

        img = draw_emotion_overlay(img, faces)

        # Grad-CAM overlay on first detected face
        if self.gradcam_on and faces:
            try:
                x, y, w, h = faces[0]['x'], faces[0]['y'], faces[0]['w'], faces[0]['h']
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (48, 48)).astype('float32') / 255.0
                inp = np.expand_dims(roi_resized, axis=(0, -1))
                heatmap = make_gradcam_heatmap(self.model, inp)
                overlay = overlay_gradcam(roi, heatmap, alpha=0.45)
                overlay_resized = cv2.resize(overlay, (w, h))
                img[y:y + h, x:x + w] = overlay_resized
            except Exception:
                pass

        # FPS overlay
        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 170), 2)

        # Face count overlay
        if len(faces) > 1:
            cv2.putText(img, f"👥 {len(faces)} faces", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        with self._lock:
            self.last_results = faces

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_emotion_bars(probs: list[float], live_update_key: str = None):
    """Renders animated 7-bar emotion probability display using HTML."""
    bars_html = ""
    for i, emotion in enumerate(EMOTIONS):
        p = probs[i] if i < len(probs) else 0
        cfg = EMOTION_CONFIG[emotion]
        width_pct = int(p * 100)
        bars_html += f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0;">
            <span style="width: 70px; font-size: 12px; color: #8B949E;">{cfg['emoji']} {emotion}</span>
            <div class="conf-track" style="flex: 1;">
                <div class="conf-fill" style="width: {width_pct}%; background: {cfg['color']};"></div>
            </div>
            <span style="width: 36px; text-align: right; font-size: 12px; color: {cfg['color']}; font-weight: 600;">{width_pct}%</span>
        </div>"""
    st.markdown(f'<div style="padding: 8px 0;">{bars_html}</div>', unsafe_allow_html=True)


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🎥 Live Camera</h1>
        <p>Real-time webcam detection with multi-face support and temporal smoothing</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_emotion_model()
    face_cascade = load_face_cascade()

    if model is None:
        return

    # ─── Sidebar Controls ──
    with st.sidebar:
        st.markdown("### 🎛️ Camera Controls")
        smoothing_on = st.toggle("⏳ Temporal Smoothing", value=st.session_state.get('smoothing_on', True),
                                 help="Smoothes predictions over last 5 frames to reduce flickering")
        gradcam_on = st.toggle("🔥 Grad-CAM Overlay", value=st.session_state.get('gradcam_on', False),
                               help="Overlays heatmap showing which facial features the CNN focuses on")
        conf_thresh = st.slider("Detection Confidence", 0.3, 0.9, 0.3, 0.05,
                                help="Minimum confidence threshold for displaying predictions")
        show_fps = st.toggle("Show FPS", value=True)

        if st.button("📸 Snapshot", use_container_width=True,
                     disabled=st.session_state.current_prediction is None):
            pred = st.session_state.current_prediction
            if pred:
                placeholder_img = Image.new('RGB', (100, 100), color='#1C2128')
                add_snapshot(placeholder_img, pred['emotion'], pred['confidence'])
                st.success(f"📸 Saved: {EMOTION_CONFIG[pred['emotion']]['emoji']} {pred['emotion']}")

        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            from utils.session_utils import reset_session
            reset_session()
            st.rerun()

    # ─── WebRTC Stream ──
    st.markdown("### 📷 Live Feed")
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,
    )

    # Update processor settings from sidebar
    if ctx.video_processor:
        ctx.video_processor.gradcam_on = gradcam_on
        ctx.video_processor.confidence_threshold = conf_thresh
        # Update smoothing window
        if smoothing_on:
            ctx.video_processor.smoother.window = 5
        else:
            ctx.video_processor.smoother.window = 1

        with ctx.video_processor._lock:
            results = list(ctx.video_processor.last_results)

        if results:
            pred = results[0]
            st.session_state.current_prediction = pred
            record_prediction(pred['emotion'], pred['confidence'], pred['probabilities'], source='live')

            # Smile alert
            if pred['emotion'] == 'Happy' and pred['confidence'] > 0.7:
                st.toast("😊 You're smiling!", icon='✅')

            # Surprise alert
            if pred['emotion'] == 'Surprise' and pred['confidence'] > 0.8:
                st.toast("😲 Surprised?", icon='⚡')

            # Face count badge
            if len(results) > 1:
                st.info(f"👥 {len(results)} faces detected")

            # ─── Dashboard ──
            col_left, col_right = st.columns([1, 1.5])

            with col_left:
                cfg = EMOTION_CONFIG[pred['emotion']]
                st.markdown(f"""
                <div class="metric-glass" style="background: {cfg['bg']}; border-color: {cfg['color']}40; text-align: center; padding: 24px;">
                    <div style="font-size: 48px;">{cfg['emoji']}</div>
                    <div style="font-size: 22px; font-weight: 700; color: {cfg['color']};">{pred['emotion']}</div>
                    <div style="font-size: 28px; font-weight: 700; color: {cfg['color']};">{pred['confidence']*100:.1f}%</div>
                    <div style="font-size: 12px; color: #8B949E; margin-top: 4px;">Dominant Emotion</div>
                </div>
                """, unsafe_allow_html=True)

                # Mood music card
                from utils.config import MOOD_MUSIC
                music = MOOD_MUSIC.get(pred['emotion'])
                if music:
                    import urllib.parse
                    spotify_url = f"https://open.spotify.com/search/{urllib.parse.quote(music['query'])}"
                    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(music['query'])}"
                    st.markdown(f"""
                    <div class="metric-glass" style="margin-top: 12px;">
                        <div style="font-size: 12px; color: #8B949E; margin-bottom: 6px;">🎵 MOOD MUSIC</div>
                        <div style="font-size: 14px; color: #E6EDF3; font-weight: 600;">{music['genre']}</div>
                        <div style="margin-top: 10px; display: flex; gap: 8px;">
                            <a href="{youtube_url}" target="_blank" style="font-size: 12px; color: #FF6B6B; text-decoration: none;">▶ YouTube</a>
                            <a href="{spotify_url}" target="_blank" style="font-size: 12px; color: #4ADE80; text-decoration: none;">● Spotify</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_right:
                st.markdown("#### 📊 All Emotions")
                render_emotion_bars(pred['probabilities'])

                # Positivity meter
                score = positivity_score(pred['probabilities'])
                pct = int((score + 1) / 2 * 100)
                st.markdown("#### 💚 Positivity Meter")
                st.markdown(f"""
                <div class="positivity-meter">
                    <div class="positivity-needle" style="left: {pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8B949E; margin-top: 4px;">
                    <span>😠 Negative</span><span>😊 Positive</span>
                </div>
                """, unsafe_allow_html=True)

                # Emotion history sparkline
                if len(st.session_state.predictions) >= 2:
                    st.markdown("#### 📈 History (last 60)")
                    recent = st.session_state.predictions[-60:]
                    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
                    y_vals = [emotion_to_num[p['emotion']] for p in recent]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(y_vals))), y=y_vals,
                        mode='lines+markers',
                        line=dict(color='#00D4AA', width=2),
                        marker=dict(color=[EMOTION_CONFIG[EMOTIONS[int(v)]]['color'] for v in y_vals], size=4),
                    ))
                    fig.update_layout(
                        yaxis=dict(tickmode='array', tickvals=list(range(7)),
                                   ticktext=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS]),
                        height=180, margin=dict(l=10, r=10, t=10, b=30),
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👤 No face detected. Ensure your face is well-lit and centered.")

    else:
        st.info("👆 Press **Start** on the video stream above to begin real-time emotion detection.")

    # ─── Snapshot Gallery ──
    if st.session_state.get('snapshots'):
        st.markdown("---")
        st.markdown("### 📸 Snapshots")
        cols = st.columns(min(4, len(st.session_state.snapshots)))
        for i, snap in enumerate(st.session_state.snapshots[-8:]):
            with cols[i % 4]:
                st.image(snap['image'], use_container_width=True)
                cfg = EMOTION_CONFIG.get(snap['emotion'], {})
                st.markdown(f"<p style='text-align: center;'>{cfg.get('emoji', '')} {snap['emotion']} {snap['confidence']*100:.0f}%</p>",
                            unsafe_allow_html=True)
