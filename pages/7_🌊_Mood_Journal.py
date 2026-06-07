"""
Page 7: 🌊 Mood Journal — Daily emotion diary with selfie scan, mood calendar, and weekly report.
"""
import streamlit as st
from datetime import datetime
from uuid import uuid4
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT, positivity_score


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🌊 Mood Journal</h1>
        <p>Track daily emotional patterns with private diary entries and calendar view</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize journal entries
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = []

    tab1, tab2, tab3 = st.tabs(["📝 New Entry", "📅 Mood Calendar", "📊 Weekly Report"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 😊 Quick Mood Check-in")
            selected_emotion = st.selectbox(
                "How are you feeling?",
                EMOTIONS,
                index=3,  # Happy default
                format_func=lambda e: f"{EMOTION_CONFIG[e]['emoji']} {e}",
            )
            intensity = st.slider("Mood Intensity", 1, 10, 5, help="1 = very mild, 10 = very intense")
            note = st.text_area("Journal Note (optional)", placeholder="What's on your mind?", max_chars=500)
            tags = st.multiselect("Tags", ["Work", "Family", "Health", "Social", "Sleep", "Exercise", "Food", "Finance", "Travel", "Other"])
            if st.button("💾 Save Entry", type="primary", use_container_width=True):
                val = EMOTION_CONFIG[selected_emotion]['valence']
                st.session_state.journal_entries.append({
                    'id': str(uuid4()),
                    'timestamp': datetime.now().isoformat(),
                    'emotion': selected_emotion,
                    'intensity': intensity,
                    'valence': val,
                    'note': note,
                    'tags': tags,
                    'source': 'manual',
                })
                st.success(f"✅ {EMOTION_CONFIG[selected_emotion]['emoji']} Mood saved!")
                st.rerun()

        with col2:
            st.markdown("#### 📸 Selfie Mood Scan")
            st.info("Upload a selfie to auto-detect your mood, or log manually.")
            uploaded_selfie = st.file_uploader("Upload a selfie", type=['jpg', 'jpeg', 'png'], key="journal_selfie")
            if uploaded_selfie:
                from utils.model_utils import load_emotion_model, load_face_cascade, predict_emotion
                from PIL import Image
                import io, cv2
                model = load_emotion_model()
                face_cascade = load_face_cascade()
                if model and face_cascade:
                    img = Image.open(io.BytesIO(uploaded_selfie.read())).convert('RGB')
                    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        roi = gray[y:y+h, x:x+w]
                        pred = predict_emotion(model, roi)
                        st.success(f"Detected: {EMOTION_CONFIG[pred['emotion']]['emoji']} {pred['emotion']} ({pred['confidence']*100:.1f}%)")
                        val = EMOTION_CONFIG[pred['emotion']]['valence']
                        if st.button("💾 Save from Selfie", use_container_width=True):
                            st.session_state.journal_entries.append({
                                'id': str(uuid4()),
                                'timestamp': datetime.now().isoformat(),
                                'emotion': pred['emotion'],
                                'intensity': min(10, int(pred['confidence']*10)),
                                'valence': val,
                                'note': 'Auto-detected from selfie',
                                'tags': [],
                                'source': 'selfie',
                            })
                            st.success("✅ Entry saved!"); st.rerun()
                    else:
                        st.error("No face detected. Please try a different image.")

    with tab2:
        entries = st.session_state.journal_entries
        if not entries:
            st.info("No journal entries yet. Use the 📝 New Entry tab to log your first mood.")
        else:
            df = pd.DataFrame(entries)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_valence = df.groupby('date')['valence'].mean().reset_index()
            daily_valence['color'] = daily_valence['valence'].apply(
                lambda v: '#4ADE80' if v > 0.3 else '#FBBF24' if v > -0.3 else '#FF6B6B'
            )
            st.markdown(f"#### 📅 Mood Calendar — {len(entries)} entries")
            for _, row in daily_valence.iterrows():
                emoji = '😊' if row['valence'] > 0.3 else '😐' if row['valence'] > -0.3 else '😟'
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;margin:4px 0;background:{row['color']}15;border-radius:8px;border-left:3px solid {row['color']};">
                    <span style="font-size:18px;">{emoji}</span>
                    <span style="flex:1;color:#E6EDF3;">{row['date']}</span>
                    <span style="color:{row['color']};font-weight:600;">{row['valence']:+.2f}</span>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        entries = st.session_state.journal_entries
        if not entries or len(entries) < 2:
            st.info("Need at least 2 entries to generate a weekly report.")
        else:
            df = pd.DataFrame(entries)
            counts = df['emotion'].value_counts()
            total = len(df)
            st.markdown("#### 📊 Weekly Mood Summary")
            for emo, cnt in counts.items():
                pct = cnt / total * 100
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:13px;">
                        <span>{EMOTION_CONFIG[emo]['emoji']} {emo}</span><span>{cnt} ({pct:.1f}%)</span>
                    </div>
                    <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{EMOTION_CONFIG[emo]['color']};"></div></div>
                </div>
                """, unsafe_allow_html=True)
            # Mood trend
            df_sorted = df.sort_values('timestamp').tail(14)
            fig = go.Figure(go.Scatter(
                x=list(range(len(df_sorted))), y=df_sorted['intensity'],
                mode='lines+markers',
                line=dict(color='#00D4AA', width=2),
                marker=dict(color=[EMOTION_CONFIG[e]['color'] for e in df_sorted['emotion']], size=8),
                text=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in df_sorted['emotion']],
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Mood Intensity Trend", height=250)
            st.plotly_chart(fig, use_container_width=True)

            # Export entries
            import json
            st.download_button("📥 Export Journal (JSON)", data=json.dumps(entries, indent=2, default=str),
                               file_name=f"mood_journal_{datetime.now().strftime('%Y%m%d')}.json",
                               mime="application/json", use_container_width=True)
