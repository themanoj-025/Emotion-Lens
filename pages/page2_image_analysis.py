"""
Page 2: 🖼️ Image Analysis — Upload images for emotion prediction
Supports batch processing, drag & drop, and export.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import os
import time
import json
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.model_utils import (
    load_model_cached, load_face_cascade, EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME
)
from utils.emotion_utils import (
    predict_from_image, draw_detection_result, generate_emotion_summary,
    compute_positivity_score, image_to_base64, anonymize_faces,
    render_mood_music_card
)
from utils.session_utils import add_prediction


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🖼️ Image Emotion Analysis</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Upload images to detect emotions — single or batch processing
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model and cascade
    model = load_model_cached()
    face_cascade = load_face_cascade()

    if model is None or face_cascade is None:
        return

    # ─── Upload Controls ──────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Upload Image(s)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Supported formats: JPG, PNG, WEBP. You can upload up to 10 images at once.",
        )
    
    with col2:
        batch_mode = st.checkbox(
            "📋 Batch Mode (process all automatically)",
            value=True,
            help="When enabled, all uploaded images are processed immediately.",
        )
        enable_auto_detect = st.checkbox(
            "🎯 Auto Face Detection",
            value=True,
            help="Automatically detect faces in images. Disable to use full image.",
        )
    
    # ─── Privacy Mode ────────────────────────────────────────
    st.markdown("---")
    privacy_col1, privacy_col2 = st.columns([1, 2])
    
    with privacy_col1:
        enable_privacy = st.checkbox(
            "🔒 Face Anonymizer", value=False,
            help="Blur detected faces to protect privacy while still detecting emotions. "
                 "Prediction runs on the original image but results display shows blurred faces.",
        )
    
    with privacy_col2:
        privacy_style = st.radio(
            "Blur Style",
            ["Gaussian Blur", "Pixelate"],
            horizontal=True,
            index=0,
            disabled=not enable_privacy,
            help="Gaussian Blur: smooth blur that masks identity. Pixelate: blocky mosaic effect.",
        )
        blur_strength = st.slider(
            "Blur Strength",
            min_value=1, max_value=5, value=3, step=1,
            disabled=not enable_privacy,
            help="Higher values = stronger blur/obfuscation. 3 is recommended for most use cases.",
            format="%d",
        )

    if not uploaded_files:
        st.info("📤 Drag & drop images or click to browse. Supports JPG, PNG, WEBP.")
        
        # Show example layout
        st.markdown(
            """
            <div style="
                border: 2px dashed #30363D;
                border-radius: 12px;
                padding: 3rem;
                text-align: center;
                color: #8B949E;
            ">
                <p style="font-size: 3rem;">🖼️</p>
                <p>Drop your images here to analyze emotions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Limit to 10 images
    if len(uploaded_files) > 10:
        st.warning(f"⚠️ Maximum 10 images supported. Showing first 10 of {len(uploaded_files)}.")
        uploaded_files = uploaded_files[:10]

    # ─── Process Images ───────────────────────────────────────
    results_data = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Read image
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Detect & predict (always on original image for accuracy)
            face_results = predict_from_image(model, face_cascade, pil_image, detect_faces=enable_auto_detect)
            
            # Start with a copy of the original image
            result_image = img_array.copy()
            
            # Apply face anonymization FIRST (before drawing labels) so faces are blurred
            # but bounding boxes and emotion labels remain readable on top
            is_anonymized = False
            if enable_privacy:
                kernel_map = {1: 31, 2: 55, 3: 81, 4: 111, 5: 151}
                k_size = kernel_map.get(blur_strength, 81)
                
                anonymize_faces(
                    result_image,
                    face_cascade=face_cascade,
                    kernel_size=(k_size, k_size),
                    pixelate=(privacy_style == "Pixelate"),
                )
                is_anonymized = True
            
            # Draw detection results ON TOP of the (possibly anonymized) image
            # This ensures labels, boxes, and confidence bars remain fully visible
            for face_result in face_results:
                result_image = draw_detection_result(result_image, face_result)
            
            # Convert back to RGB for display
            display_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Store for display
            is_fallback = any(r.get('fallback', False) for r in face_results)
            
            results_data.append({
                'filename': uploaded_file.name,
                'pil_image': pil_image,
                'display_image': display_image,
                'face_results': face_results,
                'fallback': is_fallback,
                'summary': generate_emotion_summary(face_results),
                'anonymized': is_anonymized,
                'privacy_style': privacy_style if enable_privacy else None,
            })
            
            # Add to session predictions
            for r in face_results:
                add_prediction(r['emotion'], r['confidence'], r['probabilities'])

    # ─── Display Results ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📊 Results — {len(results_data)} image(s) processed")
    
    # Summary bar
    all_emotions = []
    for d in results_data:
        for r in d['face_results']:
            all_emotions.append(r['emotion'])
    
    if all_emotions:
        emotion_counts = Counter(all_emotions)
        summary_parts = []
        for emotion, count in emotion_counts.most_common(3):
            emoji = EMOTION_CONFIG.get(emotion, {}).get('emoji', '')
            summary_parts.append(f"{emoji} {emotion}: {count}")
        st.success(f"🎯 Detected: {' | '.join(summary_parts)}")

    # ─── Individual Results ───────────────────────────────────
    for idx, data in enumerate(results_data):
        with st.expander(f"📷 {data['filename']}", expanded=(idx == 0)):
            if data['fallback']:
                st.warning("⚠️ No faces detected with Haar Cascade. Used full image for prediction.")
            
            # Show privacy badge if anonymized
            if data.get('anonymized'):
                style_label = data.get('privacy_style', 'Blur')
                col_badge, _ = st.columns([1, 3])
                with col_badge:
                    st.markdown(
                        f"""
                        <div style="
                            display: inline-block;
                            background: #2D1A0A;
                            border: 1px solid #F5A623;
                            border-radius: 20px;
                            padding: 4px 14px;
                            font-size: 0.85rem;
                            color: #F5A623;
                            margin-bottom: 0.5rem;
                        ">
                            🔒 Faces Anonymized ({style_label})
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            # Main layout: Original | Detection | Radar
            col1, col2, col3 = st.columns(3)
            
            with col1:
                col1_title = "Original Image" if not data.get('anonymized') else "📍 Original (for reference)"
                st.markdown(f"**{col1_title}**")
                st.image(data['pil_image'], use_container_width=True)
            
            with col2:
                col2_title = "🔒 Anonymized (Privacy Mode)" if data.get('anonymized') else "**Detected Faces**"
                st.markdown(f"**{col2_title}**")
                st.image(data['display_image'], use_container_width=True)
            
            with col3:
                st.markdown("**Emotion Radar**")
                if data['face_results']:
                    _render_radar_chart(data['face_results'])
            
            # Face-by-face details
            if len(data['face_results']) > 1:
                st.markdown("#### 👥 Multi-Person Analysis")
                st.info(data['summary'])
                
                # Detailed table
                rows = []
                for i, r in enumerate(data['face_results']):
                    rows.append({
                        'Face #': i + 1,
                        'Emotion': f"{EMOTION_CONFIG.get(r['emotion'], {}).get('emoji', '')} {r['emotion']}",
                        'Confidence': f"{r['confidence']*100:.1f}%",
                        'Positivity': f"{compute_positivity_score(r['probabilities']):+.2f}",
                    })
                st.table(pd.DataFrame(rows))
            
            elif data['face_results']:
                r = data['face_results'][0]
                st.markdown(f"**Detected:** {EMOTION_CONFIG.get(r['emotion'], {}).get('emoji', '')} **{r['emotion']}** with **{r['confidence']*100:.1f}%** confidence")
                
                # Positivity score
                positivity = compute_positivity_score(r['probabilities'])
                st.markdown(f"**Positivity Score:** {positivity:+.2f} {'😊' if positivity > 0 else '😟'}")
                
                # Mood music suggestion
                render_mood_music_card(r['emotion'], r['confidence'])

    # ─── Export Options ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Build CSV data
        csv_rows = []
        for d in results_data:
            for r in d['face_results']:
                row = {
                    'filename': d['filename'],
                    'emotion': r['emotion'],
                    'confidence': f"{r['confidence']*100:.1f}%",
                }
                for i, emotion in enumerate(EMOTIONS):
                    row[f'prob_{emotion}'] = f"{r['probabilities'][i]*100:.1f}%"
                csv_rows.append(row)
        
        if csv_rows:
            csv_df = pd.DataFrame(csv_rows)
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"emotion_results_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
    
    with col2:
        # JSON export
        json_data = []
        for d in results_data:
            for r in d['face_results']:
                json_data.append({
                    'filename': d['filename'],
                    'emotion': r['emotion'],
                    'confidence': r['confidence'],
                    'probabilities': {emotion: r['probabilities'][i] for i, emotion in enumerate(EMOTIONS)},
                })
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"emotion_results_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with col3:
        # Clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def _render_radar_chart(face_results):
    """Render a Plotly radar chart for emotion probabilities."""
    if not face_results:
        st.info("No data")
        return
    
    fig = go.Figure()
    
    for i, result in enumerate(face_results):
        probs = result['probabilities']
        config = EMOTION_CONFIG.get(result['emotion'], {})
        
        fig.add_trace(go.Scatterpolar(
            r=probs + [probs[0]],  # Close the radar
            theta=EMOTIONS + [EMOTIONS[0]],
            name=f"Face {i+1}",
            line=dict(color=config.get('color', '#00D4AA'), width=2),
            fill='toself',
            opacity=0.6,
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='#8B949E'),
                gridcolor='#30363D',
            ),
            angularaxis=dict(
                tickfont=dict(color='#E6EDF3', size=10),
                gridcolor='#30363D',
            ),
            bgcolor='#161B22',
        ),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=len(face_results) > 1,
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)
