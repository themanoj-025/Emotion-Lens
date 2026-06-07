"""
Page 1: 🎥 Live Camera — Real-time webcam emotion detection
Uses streamlit-webrtc for browser-native camera access with frame processing.
"""

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from utils.model_utils import (
    load_model_cached, load_face_cascade, 
    EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME
)
from utils.emotion_utils import (
    predict_emotion, draw_detection_result, apply_temporal_smoothing, 
    compute_positivity_score, preprocess_face,
    compute_gradcam, apply_gradcam_overlay,
    render_mood_music_card
)
from utils.session_utils import add_prediction, add_snapshot
import plotly.graph_objects as go
import plotly.express as px


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🎥 Live Emotion Camera</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Real-time facial emotion detection powered by CNN
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model and cascade
    model = load_model_cached()
    face_cascade = load_face_cascade()

    if model is None:
        st.error("⚠️ Model file not found. Please train a model first or place `emotion_model.h5` in the project root.")
        st.page_link("streamlit_app.py?page=Train+Model", label="➡️ Go to Train Model page", icon="🏋️")
        return

    if face_cascade is None:
        st.error("❌ Face cascade classifier failed to load. OpenCV may be misconfigured.")
        return

    # Initialize state variables
    if 'temporal_buffer' not in st.session_state:
        st.session_state.temporal_buffer = []
    if 'locked_prediction' not in st.session_state:
        st.session_state.locked_prediction = None
    if 'lock_start_time' not in st.session_state:
        st.session_state.lock_start_time = None
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    # ─── Controls ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        start_camera = st.button("▶ Start Camera", type="primary", use_container_width=True)
    
    with col2:
        snapshot_btn = st.button(
            "📸 Snapshot", use_container_width=True, 
            disabled=st.session_state.current_prediction is None,
            help="Capture the current frame with its emotion prediction to the gallery.",
        )
        # Snapshot logic: capture current prediction when button is clicked
        if snapshot_btn and st.session_state.current_prediction:
            pred = st.session_state.current_prediction
            # We store a placeholder image since we can't capture the exact frame post-hoc
            placeholder_img = Image.new('RGB', (100, 100), color='#1C2128')
            add_snapshot(placeholder_img, pred['emotion'], pred['confidence'])
            st.success(f"📸 Snapshot saved: {EMOTION_CONFIG[pred['emotion']]['emoji']} {pred['emotion']}")
    
    with col3:
        lock_btn = st.button("🔒 Lock Frame", use_container_width=True, disabled=st.session_state.current_prediction is None)
    
    with col4:
        if st.session_state.locked_prediction:
            if st.button("🔓 Unlock", use_container_width=True):
                st.session_state.locked_prediction = None
                st.session_state.lock_start_time = None

    # Grad-CAM toggle (below controls, full width)
    enable_gradcam = st.checkbox(
        "🔥 Grad-CAM Live Overlay", value=False,
        help="When enabled, overlays a Grad-CAM heatmap on detected faces showing which facial features (eyes, mouth, brow) the CNN focuses on for its prediction. May reduce FPS.",
    )
    if enable_gradcam:
        st.info("🔬 Grad-CAM active: heatmap shows where the CNN is looking — red/orange regions indicate the most influential pixels for the predicted emotion.")

    # Show lock info
    if st.session_state.locked_prediction:
        locked = st.session_state.locked_prediction
        st.info(f"🔒 Frame Locked: {EMOTION_CONFIG[locked['emotion']]['emoji']} {locked['emotion']} ({locked['confidence']*100:.1f}%) — press Unlock to release")

    # ─── WebRTC Implementation ────────────────────────────────
    st.markdown("---")
    
    # Try streamlit-webrtc first, fallback to OpenCV
    use_webrtc = st.checkbox("Use WebRTC (browser-native camera, recommended)", value=True, 
                              help="Uses browser's built-in camera API via WebRTC. More compatible than OpenCV.")

    if use_webrtc:
        _render_webrtc_camera(model, face_cascade, enable_gradcam)
    else:
        _render_opencv_fallback(model, face_cascade, enable_gradcam)

    # ─── Emotion Dashboard Area ───────────────────────────────
    st.markdown("---")
    
    current_pred = st.session_state.get('current_prediction', None)
    
    if current_pred:
        # Apply temporal smoothing to reduce flickering
        smoothed = apply_temporal_smoothing(
            st.session_state.temporal_buffer, 
            current_pred, 
            window=5
        )
        
        # Layout: Left = Dominant Emotion Card, Right = All Emotions Bar Chart
        left_col, right_col = st.columns([1, 1.5])
        
        with left_col:
            _render_dominant_emotion_card(smoothed)
            render_mood_music_card(smoothed['emotion'], smoothed['confidence'])
        
        with right_col:
            _render_emotion_bars(smoothed)
        
        # Emotion History
        st.markdown("### 📈 Emotion History (last 60 frames)")
        _render_emotion_history()

        # Smile detector alert
        if smoothed['emotion'] == 'Happy' and smoothed['confidence'] > 0.70:
            st.balloons()
            st.success("😊 **You're Smiling!** Keep it up!")
        
        # Positivity meter
        positivity = compute_positivity_score(smoothed['probabilities'])
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

    # ─── Snapshot Gallery ─────────────────────────────────────
    if st.session_state.get('snapshots'):
        st.markdown("---")
        st.markdown("### 📸 Snapshot Gallery")
        cols = st.columns(min(4, len(st.session_state.snapshots)))
        for i, snap in enumerate(st.session_state.snapshots[-8:]):
            with cols[i % 4]:
                st.image(snap['image'], use_container_width=True)
                config = EMOTION_CONFIG.get(snap['emotion'], {})
                st.markdown(
                    f"<p style='text-align: center;'>{config.get('emoji', '')} {snap['emotion']} {snap['confidence']*100:.0f}%</p>",
                    unsafe_allow_html=True,
                )


def _render_webrtc_camera(model, face_cascade, enable_gradcam=False):
    """Render the live camera feed using streamlit-webrtc."""
    st.info("📷 WebRTC mode enabled. Click 'Start' above when ready.")
    
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        import av

        class EmotionVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.face_cascade = face_cascade
                self.last_result = None
                self.frame_count = 0
                self.gradcam_enabled = enable_gradcam
                self.gradcam_frame_skip = 0
                # Store temporal buffer on the instance (thread-safe - no st.session_state access)
                self.temporal_buffer = []
            
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    try:
                        face_roi = gray[y:y+h, x:x+w]
                        emotion, confidence, probs = predict_emotion(self.model, face_roi)
                        config = EMOTION_CONFIG.get(emotion, {})
                        
                        # Draw bounding box
                        hex_color = config.get('color', '#FFFFFF').lstrip('#')
                        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                        cv2.rectangle(img, (x, y), (x + w, y + h), bgr_color, 2)
                        
                        # Label with background
                        label = f"{config.get('emoji', '')} {emotion} {confidence*100:.0f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img, (x, y - th - 15), (x + tw + 10, y), bgr_color, -1)
                        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Confidence bar
                        bar_w = int(w * confidence)
                        bar_y = y + h + 8
                        cv2.rectangle(img, (x, bar_y), (x + w, bar_y + 6), (50, 50, 50), -1)
                        cv2.rectangle(img, (x, bar_y), (x + bar_w, bar_y + 6), bgr_color, -1)

                        self.last_result = {
                            'emotion': emotion,
                            'confidence': float(confidence),
                            'probabilities': probs.tolist(),
                        }
                        self.frame_count += 1
                        
                        # Store in instance-local buffer (thread-safe vs st.session_state)
                        if len(self.temporal_buffer) >= 30:
                            self.temporal_buffer.pop(0)
                        self.temporal_buffer.append(self.last_result)
                    except Exception:
                        continue
                
                # Grad-CAM overlay on detected faces
                if self.gradcam_enabled and len(faces) > 0:
                    self.gradcam_frame_skip += 1
                    # Compute Grad-CAM every 3 frames to keep FPS reasonable
                    if self.gradcam_frame_skip % 3 == 0:
                        try:
                            # Use the first (largest) face for Grad-CAM
                            fx, fy, fw, fh = faces[0]
                            face_roi = gray[fy:fy+fh, fx:fx+fw]
                            roi_resized = cv2.resize(face_roi, (48, 48))
                            roi_input = roi_resized.astype('float32') / 255.0
                            roi_input = np.expand_dims(roi_input, axis=[0, -1])
                            
                            # Get target class
                            result = self.last_result
                            if result:
                                target = int(np.argmax(result['probabilities']))
                                heatmap = compute_gradcam(self.model, roi_input, target)
                                if heatmap is not None:
                                    apply_gradcam_overlay(img, (fx, fy, fw, fh), heatmap)
                        except Exception:
                            pass
                
                # FPS counter
                fps_label = "Grad-CAM ON" if self.gradcam_enabled else "FPS: ~15"
                cv2.putText(img, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 170), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="emotion-detection-webrtc",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Update session state with results
        if ctx.video_processor and ctx.video_processor.last_result:
            result = ctx.video_processor.last_result
            
            # Copy temporal buffer from processor instance (thread-safe)
            st.session_state.temporal_buffer = ctx.video_processor.temporal_buffer[-30:]
            
            # Record prediction periodically (every 10th frame)
            if ctx.video_processor.frame_count % 10 == 0:
                add_prediction(result['emotion'], result['confidence'], result['probabilities'])
            
            st.session_state.current_prediction = result

    except ImportError:
        st.error("❌ `streamlit-webrtc` is not installed. Falling back to OpenCV mode.")
        st.info("Install with: `pip install streamlit-webrtc av`")
        _render_opencv_fallback(model, face_cascade, enable_gradcam)
    except Exception as e:
        st.error(f"❌ WebRTC error: {e}")
        st.info("Falling back to OpenCV mode.")
        _render_opencv_fallback(model, face_cascade, enable_gradcam)


def _render_opencv_fallback(model, face_cascade, enable_gradcam=False):
    """Fallback: OpenCV-based camera (works locally)."""
    st.warning("📹 Using OpenCV fallback (WebRTC unavailable). Local webcam access may vary.")
    
    run = st.checkbox("Enable OpenCV Camera Feed", value=False)
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return
        
        st.info("Camera is running. Close the OpenCV window or uncheck to stop.")
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            results_for_frame = []
            for (x, y, w, h) in faces:
                try:
                    face_roi = gray[y:y+h, x:x+w]
                    emotion, confidence, probs = predict_emotion(model, face_roi)
                    config = EMOTION_CONFIG.get(emotion, {})
                    
                    hex_color = config.get('color', '#FFFFFF').lstrip('#')
                    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                    label = f"{config.get('emoji', '')} {emotion} {confidence*100:.0f}%"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                    
                    results_for_frame.append({
                        'emotion': emotion,
                        'confidence': float(confidence),
                        'probabilities': probs.tolist(),
                    })
                except Exception:
                    continue
            
            if results_for_frame:
                result = results_for_frame[0]
                # Populate temporal buffer
                if len(st.session_state.temporal_buffer) >= 30:
                    st.session_state.temporal_buffer.pop(0)
                st.session_state.temporal_buffer.append(result)
                
                st.session_state.current_prediction = result
                add_prediction(
                    result['emotion'],
                    result['confidence'],
                    result['probabilities'],
                )
                
                # Grad-CAM overlay on the first detected face
                if enable_gradcam and faces is not None and len(faces) > 0:
                    try:
                        fx, fy, fw, fh = faces[0]
                        face_roi = gray[fy:fy+fh, fx:fx+fw]
                        roi_resized = cv2.resize(face_roi, (48, 48))
                        roi_input = roi_resized.astype('float32') / 255.0
                        roi_input = np.expand_dims(roi_input, axis=[0, -1])
                        target = int(np.argmax(result['probabilities']))
                        heatmap = compute_gradcam(model, roi_input, target)
                        if heatmap is not None:
                            apply_gradcam_overlay(frame, (fx, fy, fw, fh), heatmap)
                    except Exception:
                        pass
            
            # FPS label
            mode_label = "OpenCV + Grad-CAM" if enable_gradcam else "OpenCV Mode"
            cv2.putText(frame, mode_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 170), 2)
            
            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
        
        cap.release()


def _render_dominant_emotion_card(result):
    """Render a large dominant emotion display card."""
    emotion = result['emotion']
    confidence = result['confidence']
    config = EMOTION_CONFIG.get(emotion, {'color': '#95A5A6', 'emoji': '❓', 'bg': '#1A1F20'})
    
    st.markdown(
        f"""
        <div style="
            background: {config['bg']};
            border: 2px solid {config['color']};
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 30px {config['color']}44;
        ">
            <div style="font-size: 4rem;">{config['emoji']}</div>
            <h2 style="color: {config['color']}; margin: 0.5rem 0; font-size: 2rem;">
                {emotion}
            </h2>
            <div style="font-size: 2.5rem; font-weight: 700; color: {config['color']};">
                {confidence*100:.1f}%
            </div>
            <p style="color: #8B949E; margin-top: 0.5rem;">Dominant Emotion</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_emotion_bars(result):
    """Render horizontal bar chart for all 7 emotions."""
    probs = result['probabilities']
    
    fig = go.Figure()
    
    for i, emotion in enumerate(EMOTIONS):
        config = EMOTION_CONFIG.get(emotion, {'color': '#95A5A6', 'emoji': '❓'})
        fig.add_trace(go.Bar(
            x=[probs[i] * 100],
            y=[emotion],
            orientation='h',
            name=emotion,
            marker_color=config['color'],
            text=[f"{config['emoji']} {probs[i]*100:.1f}%"],
            textposition='outside',
            hovertemplate=f"<b>{emotion}</b><br>Confidence: {probs[i]*100:.1f}%<extra></extra>",
        ))
    
    fig.update_layout(
        title="📊 All Emotions (Probability Distribution)",
        xaxis=dict(title="Confidence (%)", range=[0, 100], gridcolor='#30363D', 
                    tickfont=dict(color='#8B949E')),
        yaxis=dict(autorange="reversed", tickfont=dict(color='#E6EDF3', size=12)),
        height=350,
        margin=dict(l=20, r=40, t=40, b=20),
        showlegend=False,
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_emotion_history():
    """Render a sparkline/line chart of emotion history over time."""
    predictions = st.session_state.get('predictions', [])
    
    if len(predictions) < 2:
        st.info("Collecting data... showing chart when enough frames are captured.")
        return
    
    # Take last 60 predictions
    recent = predictions[-60:]
    
    # Create a numeric mapping for emotions
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    y_values = [emotion_to_num[p['emotion']] for p in recent]
    x_values = list(range(len(recent)))
    
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name='Emotion',
        line=dict(color='#00D4AA', width=2),
        marker=dict(
            color=[EMOTION_CONFIG[EMOTIONS[int(v)]]['color'] for v in y_values],
            size=6,
        ),
        hovertemplate='Frame %{x}<br>Emotion: %{customdata}<extra></extra>',
        customdata=[p['emotion'] for p in recent],
    ))
    
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            gridcolor='#30363D',
        ),
        xaxis=dict(title="Frame", gridcolor='#30363D'),
        height=250,
        margin=dict(l=20, r=20, t=10, b=30),
        hovermode='x unified',
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_positivity_gauge(positivity):
    """Render a positivity/valence gauge meter."""
    # Clamp to -1..1
    clamped = max(-1.0, min(1.0, positivity))
    
    # Color: red (negative) → yellow (neutral) → green (positive)
    if clamped > 0:
        color = f"rgb({int(46 + (210-46)*clamped)}, {int(204 + (212-204)*(1-clamped))}, {int(113 + (170-113)*clamped)})"
    else:
        color = f"rgb({int(255 + (46-255)*(-clamped))}, {int(107 + (204-107)*(-clamped))}, {int(107 + (113-107)*(-clamped))})"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=clamped * 100,
        number=dict(suffix="%", font=dict(color='#E6EDF3', size=24)),
        title=dict(text="Positivity Score", font=dict(color='#8B949E', size=14)),
        delta=dict(reference=0, font=dict(color='#8B949E')),
        gauge=dict(
            axis=dict(range=[-100, 100], tickfont=dict(color='#8B949E'),
                      tickvals=[-100, -50, 0, 50, 100],
                      ticktext=["-100", "-50", "0", "50", "100"]),
            bar=dict(color=color, thickness=0.3),
            bgcolor='#1C2128',
            borderwidth=1,
            bordercolor='#30363D',
            steps=[
                dict(range=[-100, -50], color='#2D1515'),
                dict(range=[-50, 0], color='#2D1F0A'),
                dict(range=[0, 50], color='#0A2D15'),
                dict(range=[50, 100], color='#0A3D15'),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.75,
                value=clamped * 100,
            ),
        ),
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='#1C2128',
        font=dict(color='#E6EDF3'),
    )
    
    st.plotly_chart(fig, use_container_width=True)
