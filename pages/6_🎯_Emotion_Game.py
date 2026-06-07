"""
Page 6: 🎯 Emotion Challenge Game — Interactive emotion recognition game
Two game modes: "Make This Face!" and "Guess the Emotion".
"""

import streamlit as st
import numpy as np
import cv2
import time
import random
import io
from PIL import Image
import pandas as pd

from utils.model_utils import (
    load_model_cached, load_face_cascade, EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME, predict_emotion
)


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🎯 Emotion Challenge Game</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Test your emotion recognition skills against the CNN!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model_cached()
    face_cascade = load_face_cascade()

    if model is None or face_cascade is None:
        return

    # ─── Game Stats ──────────────────────────────────────────
    game_score = st.session_state.get('game_score', 0)
    high_score = st.session_state.get('game_high_score', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Current Score", game_score)
    with col2:
        st.metric("🥇 High Score", high_score)
    with col3:
        game_history = st.session_state.get('game_history', [])
        st.metric("🎮 Games Played", len(game_history))

    # ─── Game Mode Selection ─────────────────────────────────
    st.markdown("---")
    mode = st.radio(
        "Select Game Mode",
        ["🎭 Make This Face!", "🤔 Guess the Emotion"],
        horizontal=True,
        help="'Make This Face!' uses your webcam. 'Guess the Emotion' uses uploaded images.",
    )

    if mode == "🎭 Make This Face!":
        _render_make_this_face_mode(model, face_cascade)
    else:
        _render_guess_emotion_mode(model)

    # ─── Leaderboard ─────────────────────────────────────────
    if game_history:
        st.markdown("---")
        st.markdown("### 🏆 Leaderboard")
        
        df = pd.DataFrame(game_history)
        if not df.empty and 'score' in df.columns:
            df_display = df.sort_values('score', ascending=False).head(10)
            df_display['rank'] = range(1, len(df_display) + 1)
            st.dataframe(
                df_display[['rank', 'mode', 'score', 'correct', 'total', 'timestamp']].rename(
                    columns={'rank': '#', 'mode': 'Mode', 'score': 'Score', 
                            'correct': 'Correct', 'total': 'Total', 'timestamp': 'Time'}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ─── Achievements ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏅 Achievements")
    
    achievements = {
        "Smile Master 😊": game_score >= 100,
        "Fear Factor 😨": game_score >= 50,
        "Poker Face 😐": game_score >= 25,
        "Emotion Pro 🎭": len(game_history) >= 5,
        "Perfect Score 💯": any(g.get('correct', 0) == g.get('total', 0) and g.get('total', 0) > 0 for g in game_history),
    }
    
    cols = st.columns(len(achievements))
    for idx, (badge, unlocked) in enumerate(achievements.items()):
        with cols[idx]:
            if unlocked:
                st.success(f"✅ {badge}")
            else:
                st.markdown(f"<p style='color: #30363D;'>🔒 {badge}</p>", unsafe_allow_html=True)


def _render_make_this_face_mode(model, face_cascade):
    """'Make This Face!' — user must mimic a randomly chosen emotion."""
    st.markdown("### 🎭 Make This Face!")
    
    st.info(
        "A random emotion will be shown. Make that face in front of the camera! "
        "Press **Capture** to take a snapshot and get scored. You have **10 seconds** per round."
    )

    # Initialize game state
    if 'game_target' not in st.session_state:
        st.session_state.game_target = None
    if 'game_start_time' not in st.session_state:
        st.session_state.game_start_time = None
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 0
    if 'game_last_result' not in st.session_state:
        st.session_state.game_last_result = None
    if 'game_round_over' not in st.session_state:
        st.session_state.game_round_over = False

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎮 Start New Round", type="primary", use_container_width=True):
            st.session_state.game_target = random.choice(EMOTIONS)
            st.session_state.game_start_time = time.time()
            st.session_state.game_active = True
            st.session_state.game_round += 1
            st.session_state.game_last_result = None
            st.session_state.game_round_over = False
            st.rerun()
    
    with col2:
        capture_btn = st.button("📸 Capture & Score", use_container_width=True, 
                                disabled=not st.session_state.game_active,
                                type="primary")
    
    with col3:
        if st.button("⏹️ End Game", use_container_width=True):
            st.session_state.game_active = False
            st.session_state.game_target = None
            st.session_state.game_last_result = None
            st.session_state.game_round_over = False

    if st.session_state.game_active and st.session_state.game_target:
        target = st.session_state.game_target
        target_config = EMOTION_CONFIG.get(target, {'emoji': '❓', 'color': '#00D4AA', 'bg': '#1C2128'})
        elapsed = time.time() - st.session_state.game_start_time
        remaining = max(0, 10 - elapsed)
        
        # Target display
        st.markdown(
            f"""
            <div style="
                background: {target_config['bg']};
                border: 3px solid {target_config['color']};
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 0 40px {target_config['color']}33;
            ">
                <p style="color: #8B949E; font-size: 1rem; margin: 0;">Show us...</p>
                <div style="font-size: 5rem;">{target_config['emoji']}</div>
                <h2 style="color: {target_config['color']}; font-size: 2.5rem; margin: 0.5rem 0;">
                    {target.upper()}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Timer
        timer_color = "#FF6B6B" if remaining < 3 else "#F5A623" if remaining < 6 else "#00D4AA"
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div class="game-timer" style="color: {timer_color};">{remaining:.1f}s</div>
                <p style="color: #8B949E;">Press <strong>📸 Capture & Score</strong> to check your face!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show camera preview with capture button
        st.markdown("#### 📷 Camera Preview")
        
        # Capture frame when button is pressed
        if capture_btn:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    detected_emotion = None
                    detected_conf = 0
                    
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        try:
                            _pred = predict_emotion(model, face_roi)
                            emotion, conf, probs = _pred['emotion'], _pred['confidence'], _pred['probabilities']
                            config = EMOTION_CONFIG.get(emotion, {})
                            
                            hex_color = config.get('color', '#FFFFFF').lstrip('#')
                            bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                            label = f"{config.get('emoji', '')} {emotion} {conf*100:.0f}%"
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                            
                            detected_emotion = emotion
                            detected_conf = conf
                        except Exception:
                            continue
                    
                    st.image(frame, channels="BGR", use_container_width=True)
                    
                    # Store result
                    st.session_state.game_last_result = {
                        'detected_emotion': detected_emotion,
                        'detected_conf': detected_conf,
                        'remaining': remaining,
                    }
                    st.session_state.game_round_over = True
                    
                    # Score this round
                    if detected_emotion and detected_emotion == target:
                        score = int(detected_conf * 100)
                        bonus = int(max(0, remaining * 2))
                        total_score = score + bonus
                        
                        st.balloons()
                        st.success(f"✅ **MATCH!** {EMOTION_CONFIG[target]['emoji']} {target} detected! Confidence: {detected_conf*100:.1f}%")
                        st.info(f"🏅 Score: {score} (confidence) + {bonus} (time bonus) = **{total_score}** points!")
                        
                        st.session_state.game_score += total_score
                        if st.session_state.game_score > st.session_state.game_high_score:
                            st.session_state.game_high_score = st.session_state.game_score
                        
                        st.session_state.game_history.append({
                            'mode': 'Make This Face!',
                            'target': target,
                            'detected': detected_emotion,
                            'score': total_score,
                            'correct': 1,
                            'total': 1,
                            'confidence': detected_conf,
                            'timestamp': time.strftime('%H:%M:%S'),
                        })
                    elif detected_emotion:
                        st.warning(f"Detected: {EMOTION_CONFIG.get(detected_emotion, {}).get('emoji', '')} {detected_emotion} — not a match! Need: {EMOTION_CONFIG[target]['emoji']} {target}")
                        st.session_state.game_history.append({
                            'mode': 'Make This Face!',
                            'target': target,
                            'detected': detected_emotion,
                            'score': 0,
                            'correct': 0,
                            'total': 1,
                            'confidence': detected_conf,
                            'timestamp': time.strftime('%H:%M:%S'),
                        })
                    else:
                        st.error("❌ No face detected! Make sure you're visible in the camera.")
                        st.session_state.game_history.append({
                            'mode': 'Make This Face!',
                            'target': target,
                            'detected': None,
                            'score': 0,
                            'correct': 0,
                            'total': 1,
                            'confidence': 0,
                            'timestamp': time.strftime('%H:%M:%S'),
                        })
                    
                    st.session_state.game_active = False
                    st.session_state.game_target = None
                    
                    if st.button("🔄 Next Round", type="primary", use_container_width=True):
                        st.session_state.game_target = random.choice(EMOTIONS)
                        st.session_state.game_start_time = time.time()
                        st.session_state.game_active = True
                        st.session_state.game_last_result = None
                        st.session_state.game_round_over = False
                        st.rerun()
                else:
                    st.error("❌ Could not capture frame from camera.")
                cap.release()
            else:
                st.error("❌ Could not open webcam. Check camera permissions.")
        else:
            # Just show a placeholder since no capture yet
            st.markdown(
                """
                <div style="
                    border: 2px dashed #30363D;
                    border-radius: 12px;
                    padding: 2rem;
                    text-align: center;
                    color: #8B949E;
                ">
                    <p style="font-size: 2rem;">📸</p>
                    <p>Press <strong>Capture & Score</strong> to take a photo and get scored</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Time's up check
        if remaining <= 0 and not st.session_state.game_round_over:
            st.error(f"⏰ Time's up! The target was {EMOTION_CONFIG[target]['emoji']} {target}")
            st.session_state.game_history.append({
                'mode': 'Make This Face!',
                'target': target,
                'detected': None,
                'score': 0,
                'correct': 0,
                'total': 1,
                'confidence': 0,
                'timestamp': time.strftime('%H:%M:%S'),
            })
            st.session_state.game_active = False
            st.session_state.game_target = None
            st.session_state.game_round_over = True
            
            if st.button("🔄 Try Again", type="primary", use_container_width=True):
                st.rerun()
    
    else:
        # Show last result if round just ended
        last_result = st.session_state.get('game_last_result')
        if last_result:
            st.info("💡 Round complete! Press **Start New Round** to play again.")
        
        st.markdown(
            """
            <div style="
                border: 2px dashed #30363D;
                border-radius: 16px;
                padding: 3rem;
                text-align: center;
                color: #8B949E;
            ">
                <p style="font-size: 3rem;">🎮</p>
                <p>Press <strong>Start New Round</strong> to begin!</p>
                <p style="font-size: 0.9rem;">Make the requested face, then press Capture & Score to check!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_guess_emotion_mode(model):
    """'Guess the Emotion' — user sees a face and must guess the emotion."""
    st.markdown("### 🤔 Guess the Emotion")
    
    st.info(
        "Upload a face image. Try to guess the emotion yourself, "
        "then compare with what the CNN predicts!"
    )

    uploaded_file = st.file_uploader(
        "📁 Upload a face image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a face image and try to guess the emotion before seeing the AI prediction.",
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Create a blurred preview
        img_array = np.array(pil_image.convert('RGB'))
        blurred = cv2.GaussianBlur(img_array, (21, 21), 0)
        
        st.markdown("#### 👀 Can you guess this emotion?")
        
        # Blur slider — user can gradually reveal
        reveal = st.slider(
            "Slide to reveal the face",
            min_value=0, max_value=100, value=30,
            help="Slide right to gradually reveal more of the face.",
        )
        
        if reveal < 100:
            # Show partially blurred
            blend = cv2.addWeighted(img_array, reveal / 100, blurred, 1 - reveal / 100, 0)
            st.image(blend, use_container_width=True, caption="🔍 Can you identify the emotion?")
        else:
            st.image(pil_image, use_container_width=True, caption="Full image revealed!")
        
        # User guesses
        st.markdown("#### Your Guess:")
        user_guess = st.selectbox(
            "What emotion is this person showing?",
            EMOTIONS,
            index=None,
            placeholder="Select an emotion...",
            help="Try to guess before revealing the AI's answer!",
        )

        if user_guess:
            # Get model prediction
            from utils.emotion_utils import predict_from_image
            face_cascade = load_face_cascade()
            
            with st.spinner("🤖 AI is analyzing..."):
                results = predict_from_image(model, face_cascade, pil_image, detect_faces=True)
            
            if results:
                ai_emotion = results[0]['emotion']
                ai_conf = results[0]['confidence']
                ai_config = EMOTION_CONFIG.get(ai_emotion, {'emoji': '❓', 'color': '#95A5A6'})
                
                # Compare
                is_correct = user_guess == ai_emotion
                
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background: {'#0A2D15' if is_correct else '#2D1515'};
                            border: 2px solid {'#2ECC71' if is_correct else '#FF6B6B'};
                            border-radius: 16px;
                            padding: 1.5rem;
                            text-align: center;
                        ">
                            <p style="color: #8B949E; margin: 0;">Your Guess</p>
                            <p style="font-size: 2rem; margin: 0.5rem 0;">
                                {EMOTION_CONFIG.get(user_guess, {}).get('emoji', '')}
                            </p>
                            <h3 style="color: {'#2ECC71' if is_correct else '#FF6B6B'}; margin: 0;">
                                {user_guess}
                            </h3>
                            <p style="color: {'#2ECC71' if is_correct else '#FF6B6B'};">
                                {'✅ Correct!' if is_correct else '❌ Not quite...'}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background: {ai_config.get('bg', '#1C2128')};
                            border: 2px solid {ai_config.get('color', '#00D4AA')};
                            border-radius: 16px;
                            padding: 1.5rem;
                            text-align: center;
                        ">
                            <p style="color: #8B949E; margin: 0;">AI Prediction</p>
                            <p style="font-size: 2rem; margin: 0.5rem 0;">
                                {ai_config.get('emoji', '')}
                            </p>
                            <h3 style="color: {ai_config.get('color', '#00D4AA')}; margin: 0;">
                                {ai_emotion}
                            </h3>
                            <p style="color: {ai_config.get('color', '#00D4AA')};">
                                Confidence: {ai_conf*100:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # Score
                if is_correct:
                    score = int(ai_conf * 100)
                    st.session_state.game_score += score
                    if st.session_state.game_score > st.session_state.game_high_score:
                        st.session_state.game_high_score = st.session_state.game_score
                    
                    st.success(f"🎉 You scored **{score}** points! Total: **{st.session_state.game_score}**")
                
                else:
                    st.info(f"💡 The AI detected {ai_config['emoji']} {ai_emotion}. Look at the {ai_emotion.lower()} facial features — the mouth, eyes, and brow area.")
                
                # Record game
                if 'game_history' not in st.session_state:
                    st.session_state.game_history = []
                st.session_state.game_history.append({
                    'mode': 'Guess the Emotion',
                    'target': ai_emotion,
                    'detected': user_guess,
                    'score': int(ai_conf * 100) if is_correct else 0,
                    'correct': 1 if is_correct else 0,
                    'total': 1,
                    'confidence': ai_conf,
                    'timestamp': time.strftime('%H:%M:%S'),
                })
            else:
                st.error("❌ No face detected in the image. Please try a different image.")
