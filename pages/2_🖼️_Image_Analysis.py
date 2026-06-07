"""
Page 2: 🖼️ Image Analysis — Batch image upload + analysis.
Processes up to 10 images with face detection, Grad-CAM heatmaps,
emotion radar charts, CSV/JSON export, and face anonymization.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import time
import json
from collections import Counter
import plotly.graph_objects as go
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT, positivity_score
from utils.model_utils import (
    load_emotion_model, load_face_cascade, predict_emotion, preprocess_roi, hex_to_bgr
)
from utils.session_utils import record_prediction
from utils.gradcam_utils import make_gradcam_heatmap, overlay_gradcam


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🖼️ Image Analysis</h1>
        <p>Batch process up to 10 images with Grad-CAM heatmap visualization</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_emotion_model()
    face_cascade = load_face_cascade()
    if model is None:
        return

    # ─── Upload Controls ──
    uploaded_files = st.file_uploader(
        "📁 Upload Image(s)", type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        accept_multiple_files=True,
        help="Supported formats: JPG, PNG, WEBP, BMP. Up to 10 images at once."
    )

    col1, col2 = st.columns(2)
    with col1:
        enable_gradcam = st.checkbox("🔥 Show Grad-CAM Heatmaps", value=True,
                                     help="Overlay heatmaps showing what the CNN focuses on")
    with col2:
        enable_privacy = st.checkbox("🔒 Face Anonymizer", value=False,
                                     help="Blur faces for privacy while still showing emotion detection")
        if enable_privacy:
            privacy_style = st.radio("Style", ["Gaussian Blur", "Pixelate"], horizontal=True, index=0)

    if not uploaded_files:
        st.info("📤 Drop images here or click to browse. Supports JPG, PNG, WEBP, BMP.")
        return

    if len(uploaded_files) > 10:
        st.warning(f"⚠️ Max 10 images. Showing first 10 of {len(uploaded_files)}.")
        uploaded_files = uploaded_files[:10]

    # ─── Process All Images ──
    all_results = []
    for idx, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_array = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            face_results = []
            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                pred = predict_emotion(model, roi)
                pred.update({'x': x, 'y': y, 'w': w, 'h': h})
                face_results.append(pred)
                record_prediction(pred['emotion'], pred['confidence'], pred['probabilities'], source='image')

            # Anonymize faces if enabled
            display_img = img_array.copy()
            if enable_privacy and face_results:
                for fr in face_results:
                    x, y, w, h = fr['x'], fr['y'], fr['w'], fr['h']
                    roi_face = display_img[y:y + h, x:x + w]
                    if privacy_style == "Gaussian Blur":
                        k = max(21, (w // 5) * 2 + 1)
                        display_img[y:y + h, x:x + w] = cv2.GaussianBlur(roi_face, (k, k), 30)
                    else:
                        small = cv2.resize(roi_face, (max(4, w // 15), max(4, h // 15)))
                        display_img[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            # Draw bounding boxes
            for fr in face_results:
                x, y, w, h = fr['x'], fr['y'], fr['w'], fr['h']
                color = hex_to_bgr(EMOTION_CONFIG[fr['emotion']]['color'])
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                label = f"{EMOTION_CONFIG[fr['emotion']]['emoji']} {fr['emotion']} {fr['confidence']*100:.0f}%"
                cv2.putText(display_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Grad-CAM on first face
            gradcam_overlay = None
            if enable_gradcam and face_results:
                try:
                    fr = face_results[0]
                    roi = gray[fr['y']:fr['y'] + fr['h'], fr['x']:fr['x'] + fr['w']]
                    roi_r = cv2.resize(roi, (48, 48)).astype('float32') / 255.0
                    inp = np.expand_dims(roi_r, axis=(0, -1))
                    heatmap = make_gradcam_heatmap(model, inp)
                    overlay_img = overlay_gradcam(roi, heatmap, alpha=0.45)
                    gradcam_overlay = cv2.resize(overlay_img, (fr['w'], fr['h']))
                except Exception:
                    pass

            all_results.append({
                'filename': uploaded_file.name,
                'pil_image': pil_image,
                'display_img': cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                'face_results': face_results,
                'gradcam_overlay': gradcam_overlay,
            })

    # ─── Display Results Grid ──
    st.markdown(f"### 📊 Results — {len(all_results)} image(s)")
    all_emotions = [r['emotion'] for d in all_results for r in d['face_results']]
    if all_emotions:
        counts = Counter(all_emotions)
        summary = ' | '.join(f"{EMOTION_CONFIG[e]['emoji']} {e}: {c}" for e, c in counts.most_common(3))
        st.success(f"🎯 Detected: {summary}")

    for idx, data in enumerate(all_results):
        with st.expander(f"📷 {data['filename']}", expanded=(idx == 0)):
            if not data['face_results']:
                st.warning("👤 No face detected in this image.")
                continue

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original**")
                st.image(data['pil_image'], use_container_width=True)
            with col2:
                st.markdown("**Detection**")
                st.image(data['display_img'], use_container_width=True)
            with col3:
                if enable_gradcam and data.get('gradcam_overlay') is not None:
                    st.markdown("**🔥 Grad-CAM**")
                    overlay_rgb = cv2.cvtColor(data['gradcam_overlay'], cv2.COLOR_BGR2RGB)
                    st.image(overlay_rgb, use_container_width=True)
                else:
                    st.markdown("**Emotion Radar**")
                    fr = data['face_results'][0]
                    fig = go.Figure(go.Scatterpolar(
                        r=fr['probabilities'] + [fr['probabilities'][0]],
                        theta=EMOTIONS + [EMOTIONS[0]],
                        fill='toself', fillcolor='rgba(0,212,170,0.15)',
                        line=dict(color='#00D4AA', width=2),
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, polar=dict(
                        bgcolor='#161B22',
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(139,148,158,0.15)'),
                        angularaxis=dict(gridcolor='rgba(139,148,158,0.15)'),
                    ), height=260, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            # Details
            fr = data['face_results'][0]
            st.markdown(f"**Detected:** {EMOTION_CONFIG[fr['emotion']]['emoji']} **{fr['emotion']}** — {fr['confidence']*100:.1f}% confidence")
            pos = positivity_score(fr['probabilities'])
            st.markdown(f"**Positivity Score:** {pos:+.2f} {'😊' if pos > 0 else '😟'}")

    # ─── Export ──
    st.markdown("---")
    st.markdown("### 📥 Export")
    col1, col2 = st.columns(2)
    csv_rows = []
    for d in all_results:
        for r in d['face_results']:
            row = {'filename': d['filename'], 'emotion': r['emotion'],
                   'confidence': f"{r['confidence']*100:.1f}%"}
            for i, e in enumerate(EMOTIONS):
                row[f'prob_{e}'] = f"{r['probabilities'][i]*100:.1f}%"
            csv_rows.append(row)
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_buf = io.StringIO()
        csv_df.to_csv(csv_buf, index=False)
        with col1:
            st.download_button("📥 Download CSV", data=csv_buf.getvalue(),
                               file_name=f"emotion_results_{int(time.time())}.csv",
                               mime="text/csv", use_container_width=True, type="primary")
        json_data = [{'filename': d['filename'], 'emotion': r['emotion'], 'confidence': r['confidence'],
                       'probabilities': {e: r['probabilities'][i] for i, e in enumerate(EMOTIONS)}}
                     for d in all_results for r in d['face_results']]
        with col2:
            st.download_button("📥 Download JSON", data=json.dumps(json_data, indent=2),
                               file_name=f"emotion_results_{int(time.time())}.json",
                               mime="application/json", use_container_width=True)
