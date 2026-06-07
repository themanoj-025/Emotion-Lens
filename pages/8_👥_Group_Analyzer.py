"""
Page 8: 👥 Group Analyzer — Multi-person crowd emotion analysis with face grid,
group summary, outlier detection, and privacy mode.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT, positivity_score
from utils.model_utils import load_emotion_model, load_face_cascade, predict_emotion, hex_to_bgr
from utils.emotion_utils import anonymize_faces


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>👥 Group Analyzer</h1>
        <p>Analyze group photos and produce aggregate crowd emotion summaries</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_emotion_model()
    face_cascade = load_face_cascade()
    if model is None:
        return

    uploaded_file = st.file_uploader("📁 Upload a group photo", type=['jpg', 'jpeg', 'png', 'webp'],
                                     help="Upload a photo with one or more faces for group emotion analysis.")
    privacy_mode = st.checkbox("🔒 Face Anonymizer", value=False,
                               help="Blur all face regions for privacy while still showing emotion stats")

    if not uploaded_file:
        st.info("📤 Upload a group photo to analyze the emotional dynamics of the crowd.")
        return

    # Process image
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("👤 No faces detected. Try a different image with clearer faces.")
        return

    # Detect emotions for each face
    face_results = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        pred = predict_emotion(model, roi)
        pred['bbox'] = (x, y, w, h)
        face_results.append(pred)

    # Anonymize if needed
    display_img = img_array.copy()
    if privacy_mode:
        display_img = anonymize_faces(display_img, face_cascade)

    # Draw bounding boxes on display image
    for fr in face_results:
        x, y, w, h = fr['bbox']
        color = hex_to_bgr(EMOTION_CONFIG[fr['emotion']]['color'])
        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
        label = f"{EMOTION_CONFIG[fr['emotion']]['emoji']} {fr['emotion']}"
        cv2.putText(display_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ─── Display Results ──
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"👥 {len(faces)} faces detected")

    with col2:
        # Group summary
        emotion_counts = Counter(fr['emotion'] for fr in face_results)
        total = len(face_results)
        st.markdown(f"### 👥 {total} people")
        st.markdown("#### Emotion Breakdown")
        for emo, cnt in emotion_counts.most_common():
            pct = cnt / total * 100
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{EMOTION_CONFIG[emo]['emoji']} {emo}</span><span>{cnt} ({pct:.0f}%)</span>
                </div>
                <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{EMOTION_CONFIG[emo]['color']};"></div></div>
            </div>
            """, unsafe_allow_html=True)

        # Group mood
        avg_pos = np.mean([positivity_score(fr['probabilities']) for fr in face_results])
        mood = "😊 Positive" if avg_pos > 0.3 else "😟 Negative" if avg_pos < -0.3 else "😐 Neutral"
        energy = round(np.mean([EMOTION_CONFIG[fr['emotion']]['arousal'] for fr in face_results]) * 100)
        st.markdown(f"""
        <div class="metric-glass" style="text-align:center;padding:16px;">
            <div style="font-size:14px;color:#8B949E;">Group Mood</div>
            <div style="font-size:24px;font-weight:700;">{mood}</div>
            <div style="font-size:12px;color:#8B949E;">Energy: {energy}% · Positivity: {avg_pos:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Outlier detection
        dominant_emo = emotion_counts.most_common(1)[0][0]
        outliers = [fr for fr in face_results if fr['emotion'] != dominant_emo]
        if outliers:
            st.markdown("#### ⚡ Outliers Detected")
            for fr in outliers:
                st.markdown(f"• {EMOTION_CONFIG[fr['emotion']]['emoji']} {fr['emotion']} — different from group ({dominant_emo})")

    # ─── Face Grid ──
    st.markdown("---")
    st.markdown("#### 🖼️ Individual Faces")
    cols = st.columns(min(4, len(face_results)))
    for i, fr in enumerate(face_results):
        with cols[i % 4]:
            x, y, w, h = fr['bbox']
            face_crop = pil_image.crop((x, y, x+w, y+h))
            st.image(face_crop, use_container_width=True)
            st.markdown(f"<p style='text-align:center;color:{EMOTION_CONFIG[fr['emotion']]['color']};font-weight:600;'>{EMOTION_CONFIG[fr['emotion']]['emoji']} {fr['emotion']}<br>{fr['confidence']*100:.0f}%</p>", unsafe_allow_html=True)

    # ─── Emotion Radar (aggregate) ──
    st.markdown("#### 📊 Aggregate Emotion Profile")
    avg_probs = np.mean([fr['probabilities'] for fr in face_results], axis=0)
    fig = go.Figure(go.Scatterpolar(
        r=list(avg_probs) + [avg_probs[0]],
        theta=EMOTIONS + [EMOTIONS[0]],
        fill='toself', fillcolor='rgba(0,212,170,0.15)',
        line=dict(color='#00D4AA', width=2),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, polar=dict(
        bgcolor='#161B22',
        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(139,148,158,0.15)'),
        angularaxis=dict(gridcolor='rgba(139,148,158,0.15)'),
    ), height=350)
    st.plotly_chart(fig, use_container_width=True)
