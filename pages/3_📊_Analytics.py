"""
Page 3: 📊 Analytics Dashboard — Session statistics & emotion history charts
Provides comprehensive visualization of all predictions made during the session.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json
from collections import Counter

from utils.model_utils import EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME
from utils.session_utils import (
    get_prediction_dataframe, get_emotion_distribution,
    export_predictions_csv, export_predictions_json,
    format_session_duration, reset_session,
)
from utils.emotion_utils import compute_positivity_score, render_mood_music_card


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>📊 Analytics Dashboard</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Session statistics, emotion distribution, and trends
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_prediction_dataframe()

    # ─── Top Metrics ──────────────────────────────────────────
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_preds = len(st.session_state.get('predictions', []))
    
    with col1:
        st.metric("Total Predictions", total_preds)
    
    with col2:
        if df.empty:
            st.metric("Session Duration", "00:00:00")
        else:
            st.metric("Session Duration", format_session_duration())
    
    with col3:
        if not df.empty:
            most_common = df['emotion'].mode()
            top_emotion = most_common.iloc[0] if not most_common.empty else "—"
            emoji = EMOTION_CONFIG.get(top_emotion, {}).get('emoji', '')
            st.metric("Most Detected", f"{emoji} {top_emotion}")
        else:
            st.metric("Most Detected", "—")
    
    with col4:
        if not df.empty:
            avg_conf = df['confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        else:
            st.metric("Avg Confidence", "—%")

    if df.empty:
        st.info("📭 No predictions recorded yet. Use the **Live Camera** or **Image Analysis** pages to collect data.")
        return

    # ─── Charts ───────────────────────────────────────────────
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Distribution", "📈 Over Time", "📊 Confidence", 
        "🔥 Heatmap", "💚 Positivity", "🎬 Timeline Replay"
    ])
    
    with tab1:
        _render_distribution_chart(df)
    
    with tab2:
        _render_timeline_chart(df)
    
    with tab3:
        _render_confidence_chart(df)
    
    with tab4:
        _render_heatmap(df)
    
    with tab5:
        _render_positivity_analysis(df)
    
    with tab6:
        _render_timeline_replay(df)

    # ─── Raw Data & Export ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Raw Data & Export")
    
    with st.expander("📄 View Raw Prediction Data"):
        display_df = df.drop(columns=['probabilities'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = export_predictions_csv()
        if csv_data:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
    
    with col2:
        json_data = export_predictions_json()
        if json_data:
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
    
    with col3:
        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()


def _render_distribution_chart(df):
    """Render emotion distribution pie and bar charts."""
    st.markdown("#### Emotion Distribution")
    
    dist = df['emotion'].value_counts()
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in dist.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in dist.index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=emoji_labels,
            values=dist.values,
            marker=dict(colors=colors, line=dict(color='#161B22', width=2)),
            textinfo='label+percent',
            textfont=dict(color='#E6EDF3', size=12),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        )])
        fig_pie.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = go.Figure(data=[go.Bar(
            x=emoji_labels,
            y=dist.values,
            marker_color=colors,
            text=dist.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
        )])
        fig_bar.update_layout(
            title="Count per Emotion",
            yaxis=dict(title="Count", gridcolor='#30363D'),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_timeline_chart(df):
    """Render emotion over time line chart."""
    st.markdown("#### Emotion Timeline")
    
    if 'timestamp' not in df.columns:
        st.warning("Timestamp data not available.")
        return
    
    df_time = df.copy()
    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
    df_time = df_time.sort_values('timestamp')
    
    # Numeric encoding for emotions
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    df_time['emotion_num'] = df_time['emotion'].map(emotion_to_num)
    
    # Create scatter trace with color-coded markers
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in df_time['emotion']]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in df_time['emotion']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_time['timestamp'],
        y=df_time['emotion_num'],
        mode='lines+markers',
        line=dict(color='#00D4AA', width=2),
        marker=dict(
            color=colors,
            size=8,
            line=dict(color='#161B22', width=1),
        ),
        text=emoji_labels,
        hovertemplate='<b>%{text}</b><br>Time: %{x|%H:%M:%S}<br>Confidence: %{customdata:.1f}%<extra></extra>',
        customdata=df_time['confidence'] * 100,
    ))
    
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            gridcolor='#30363D',
            title="Emotion",
        ),
        xaxis=dict(
            title="Time",
            gridcolor='#30363D',
            tickformat='%H:%M:%S',
        ),
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=30, b=30),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_chart(df):
    """Render average confidence per emotion bar chart."""
    st.markdown("#### Average Confidence per Emotion")
    
    avg_conf = df.groupby('emotion')['confidence'].mean() * 100
    avg_conf = avg_conf.reindex(EMOTIONS, fill_value=0)
    
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in avg_conf.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in avg_conf.index]
    
    fig = go.Figure(data=[go.Bar(
        x=emoji_labels,
        y=avg_conf.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in avg_conf.values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Confidence: %{y:.1f}%<extra></extra>',
    )])
    
    fig.update_layout(
        yaxis=dict(title="Average Confidence (%)", range=[0, 100], gridcolor='#30363D'),
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(df):
    """Render a confidence heatmap showing prediction patterns."""
    st.markdown("#### Confidence Heatmap (Last 100 Predictions)")
    
    recent = df.tail(100).copy()
    if len(recent) < 2:
        st.info("Need more data points to generate heatmap.")
        return
    
    # Create a transition matrix
    emotion_order = EMOTIONS
    n = len(emotion_order)
    transition_matrix = np.zeros((n, n))
    
    for i in range(len(recent) - 1):
        curr = recent.iloc[i]['emotion']
        next_e = recent.iloc[i + 1]['emotion']
        if curr in emotion_order and next_e in emotion_order:
            transition_matrix[emotion_order.index(curr)][emotion_order.index(next_e)] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition_matrix = transition_matrix / row_sums
    
    labels = [f"{EMOTION_CONFIG[e]['emoji']} {e[:3]}" for e in emotion_order]
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        text=np.round(transition_matrix, 2),
        texttemplate='%{text:.0%}',
        textfont=dict(size=10, color='#E6EDF3'),
        hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.1%}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Emotion Transition Probabilities",
        xaxis=dict(title="Next Emotion", tickfont=dict(size=10)),
        yaxis=dict(title="Current Emotion", tickfont=dict(size=10)),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_positivity_analysis(df):
    """Render positivity/valence score trends."""
    st.markdown("#### Positivity Score Over Time")
    
    if len(df) < 2:
        st.info("Need more data to show positivity trend.")
        return
    
    # Compute positivity for each prediction
    positivity_scores = []
    for _, row in df.iterrows():
        probs = row.get('probabilities')
        if probs and len(probs) == 7:
            score = compute_positivity_score(probs)
            positivity_scores.append(score)
        else:
            positivity_scores.append(0)
    
    df_viz = df.tail(100).copy()
    scores = positivity_scores[-100:]
    
    # Color mapping
    colors = ['#FF6B6B' if s < -0.3 else '#F5A623' if s < 0.3 else '#2ECC71' for s in scores]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode='lines+markers',
        name='Positivity Score',
        line=dict(color='#00D4AA', width=2),
        marker=dict(color=colors, size=5),
        hovertemplate='Frame %{x}<br>Score: %{y:+.2f}<extra></extra>',
    ), secondary_y=False)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#8B949E", opacity=0.5)
    
    # Add rolling average
    if len(scores) > 5:
        window = min(10, len(scores) // 2)
        rolling_avg = pd.Series(scores).rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=rolling_avg,
            mode='lines',
            name=f'{window}-frame Average',
            line=dict(color='#FFFFFF', width=2, dash='dot'),
            hovertemplate='Frame %{x}<br>Avg Score: %{y:+.2f}<extra></extra>',
        ), secondary_y=False)
    
    fig.update_layout(
        yaxis=dict(title="Positivity Score", range=[-1.1, 1.1], gridcolor='#30363D',
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0 😟", "-0.5", "0 😐", "0.5", "1.0 😊"]),
        xaxis=dict(title="Frame (recent)", gridcolor='#30363D'),
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=30, b=30),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    avg_pos = np.mean(positivity_scores) if positivity_scores else 0
    if avg_pos > 0.3:
        mood_label = "😊 Positive"
        mood_color = "#2ECC71"
    elif avg_pos < -0.3:
        mood_label = "😟 Negative"
        mood_color = "#FF6B6B"
    else:
        mood_label = "😐 Neutral"
        mood_color = "#F5A623"
    
    st.markdown(
        f"""
        <div style="
            background: #1C2128;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 0.5rem;
        ">
            <h3 style="color: #8B949E; margin: 0; font-size: 0.9rem; text-transform: uppercase;">
                Session Mood Summary
            </h3>
            <p style="font-size: 2rem; margin: 0.5rem 0; color: {mood_color};">
                {mood_label}
            </p>
            <p style="color: #8B949E;">
                Average Positivity Score: <span style="color: {mood_color}; font-weight: 700;">{avg_pos:+.3f}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Show mood music suggestion for the dominant session mood
    top_emotion = df['emotion'].mode().iloc[0] if not df.empty else None
    if top_emotion:
        render_mood_music_card(top_emotion)


def _render_timeline_replay(df):
    """
    🎬 Emotion Timeline Replay — Record a segment of predictions and play it back
    as an animated chart showing the emotional journey over time.
    """
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color: #00D4AA; margin: 0;">🎬 Emotion Timeline Replay</h3>
            <p style="color: #8B949E; font-size: 0.9rem;">
                Record a segment of predictions and replay your emotional journey
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ─── Controls ────────────────────────────────────────────
    rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 1])
    
    currently_recording = st.session_state.timeline_recording
    
    with rec_col1:
        record_duration = st.slider(
            "Recording Duration (seconds)",
            min_value=10, max_value=120, value=60, step=10,
            disabled=currently_recording,
            help="How long to record predictions. The dashboard will collect all predictions made during this window.",
        )
    
    with rec_col2:
        if not currently_recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True,
                         disabled=df.empty):
                # Start recording: mark the current prediction count as the start
                st.session_state.timeline_recording = True
                st.session_state.timeline_start_idx = len(st.session_state.get('predictions', []))
                st.session_state.timeline_recording_start = datetime.now()
                st.session_state.timeline_recording_end = record_duration
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                _save_timeline_recording()
                st.rerun()
    
    with rec_col3:
        if st.session_state.timeline_recordings:
            recording_names = [r['name'] for r in st.session_state.timeline_recordings]
            selected_recording = st.selectbox(
                "Previous Recordings",
                options=list(range(len(recording_names))),
                format_func=lambda i: recording_names[i],
                index=len(recording_names) - 1,
                help="Select a previous recording to replay.",
            )
        else:
            selected_recording = None
            st.markdown("<p style='color:#8B949E; padding-top:1.8rem;'>No recordings yet</p>", unsafe_allow_html=True)
    
    # ─── Recording Progress ──────────────────────────────────
    if currently_recording:
        elapsed = (datetime.now() - st.session_state.timeline_recording_start).total_seconds()
        remaining = max(0, st.session_state.timeline_recording_end - elapsed)
        progress = min(1.0, elapsed / st.session_state.timeline_recording_end)
        
        st.progress(progress)
        
        # Show live recording stats
        current_count = len(st.session_state.get('predictions', []))
        new_preds = current_count - st.session_state.timeline_start_idx
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("⏱️ Time Remaining", f"{remaining:.0f}s")
        col_b.metric("📊 Predictions Recorded", new_preds)
        col_c.metric("📈 Progress", f"{progress*100:.0f}%")
        
        # Auto-stop when duration reached
        if elapsed >= st.session_state.timeline_recording_end:
            _save_timeline_recording()
            st.rerun()
        
        st.info(f"🔴 Recording in progress... Auto-stop in {remaining:.0f} seconds. Make predictions using the Live Camera or Image Analysis pages.")
        
        # Show a placeholder chart with what's been recorded so far
        current_preds = st.session_state.get('predictions', [])
        new_slice = current_preds[st.session_state.timeline_start_idx:]
        if len(new_slice) >= 2:
            _render_live_recording_chart(new_slice)
        
        return  # Don't show playback while recording
    
    # ─── Playback Mode ───────────────────────────────────────
    if selected_recording is not None and st.session_state.timeline_recordings:
        recording = st.session_state.timeline_recordings[selected_recording]
        _render_playback_view(recording)
    else:
        if df.empty:
            st.info("📭 No predictions yet. Use the **Live Camera** or **Image Analysis** pages to collect data first.")
        else:
            st.info("👆 Press **🔴 Start Recording** to begin capturing predictions. The replay will appear here once recording completes.")


def _save_timeline_recording():
    """Save the current recording to session state."""
    predictions = st.session_state.get('predictions', [])
    start_idx = st.session_state.timeline_start_idx
    recording_slice = predictions[start_idx:]
    
    if not recording_slice:
        st.session_state.timeline_recording = False
        return
    
    start_time = st.session_state.get('timeline_recording_start', datetime.now())
    duration = st.session_state.get('timeline_recording_end', 60)
    
    timestamp_str = start_time.strftime('%H:%M:%S')
    emotion_counts = {}
    for p in recording_slice:
        e = p['emotion']
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    top_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else '—'
    emoji = EMOTION_CONFIG.get(top_emotion, {}).get('emoji', '')
    
    st.session_state.timeline_recordings.append({
        'name': f"{timestamp_str} — {emoji} {top_emotion} ({len(recording_slice)} preds)",
        'predictions': recording_slice,
        'start_time': start_time.isoformat(),
        'duration': duration,
        'count': len(recording_slice),
        'top_emotion': top_emotion,
    })
    st.session_state.timeline_recording = False


def _render_live_recording_chart(predictions_slice):
    """Render a live preview of the current recording."""
    df_rec = pd.DataFrame(predictions_slice)
    if 'timestamp' in df_rec.columns:
        df_rec['timestamp'] = pd.to_datetime(df_rec['timestamp'])
        df_rec = df_rec.sort_values('timestamp')
    
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    df_rec['emotion_num'] = df_rec['emotion'].map(emotion_to_num)
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in df_rec['emotion']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df_rec))),
        y=df_rec['emotion_num'],
        mode='lines+markers',
        line=dict(color='#00D4AA', width=2),
        marker=dict(color=colors, size=6),
        text=[f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in df_rec['emotion']],
        hovertemplate='<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>',
        customdata=df_rec['confidence'] * 100,
    ))
    fig.update_layout(
        title="🔴 Live Recording Preview",
        yaxis=dict(tickmode='array', tickvals=list(range(7)),
                   ticktext=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
                   gridcolor='#30363D'),
        xaxis=dict(title="Frame (since recording started)", gridcolor='#30363D'),
        height=300,
        margin=dict(l=20, r=20, t=40, b=30),
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_playback_view(recording):
    """Render the animated playback view for a completed recording."""
    preds = recording['predictions']
    count = recording['count']
    
    df_rec = pd.DataFrame(preds)
    if 'timestamp' in df_rec.columns:
        df_rec['timestamp'] = pd.to_datetime(df_rec['timestamp'])
        df_rec = df_rec.sort_values('timestamp')
    
    # ─── Recording Summary ───────────────────────────────────
    st.markdown("---")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("📊 Predictions", count)
    with col_s2:
        st.metric("⏱️ Duration", f"{recording['duration']}s")
    with col_s3:
        top_emoji = EMOTION_CONFIG.get(recording['top_emotion'], {}).get('emoji', '')
        st.metric("🎯 Dominant", f"{top_emoji} {recording['top_emotion']}")
    with col_s4:
        # Compute average positivity
        positivity_scores = []
        for _, row in df_rec.iterrows():
            probs = row.get('probabilities')
            if probs and len(probs) == 7:
                positivity_scores.append(compute_positivity_score(probs))
        avg_pos = np.mean(positivity_scores) if positivity_scores else 0
        mood = "😊" if avg_pos > 0.3 else "😟" if avg_pos < -0.3 else "😐"
        st.metric("💚 Mood", f"{mood} {avg_pos:+.2f}")
    
    # ─── Playback Speed Control ──────────────────────────────
    st.markdown("---")
    play_col1, play_col2 = st.columns([1, 3])
    
    with play_col1:
        playback_speed = st.select_slider(
            "Playback Speed",
            options=["0.25x", "0.5x", "1x", "2x", "4x"],
            value="1x",
            help="Controls how fast the animation plays. 1x = real-time speed.",
        )
        speed_mult = float(playback_speed.replace('x', ''))
    
    with play_col2:
        st.markdown("<p style='padding-top:0.5rem;'></p>", unsafe_allow_html=True)
        play_btn_col, restart_col = st.columns(2)
        with play_btn_col:
            play_button = st.button("▶️ Play Animation", type="primary", use_container_width=True)
        with restart_col:
            restart_button = st.button("⏹️ Reset", use_container_width=True)
    
    # ─── Animated Chart with Frames ──────────────────────────
    if count < 2:
        st.info("Need at least 2 data points for playback.")
        return
    
    # Compute display data
    n_points = min(count, 300)  # Cap at 300 for performance
    step = max(1, count // n_points)
    plot_data = df_rec.iloc[::step].head(n_points).reset_index(drop=True)
    n_frames = len(plot_data)
    
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    plot_data['emotion_num'] = plot_data['emotion'].map(emotion_to_num)
    
    # Create the base figure (empty initially)
    fig = go.Figure()
    
    # Add the full trace as a faint background reference
    fig.add_trace(go.Scatter(
        x=list(range(n_frames)),
        y=plot_data['emotion_num'],
        mode='lines',
        line=dict(color='#30363D', width=1, dash='dot'),
        name='Full path',
        hovertemplate='<extra></extra>',
        showlegend=False,
    ))
    
    # Add the animated trace (starts empty, filled by frames)
    colors_full = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in plot_data['emotion']]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in plot_data['emotion']]
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[plot_data['emotion_num'].iloc[0]],
        mode='lines+markers',
        line=dict(color='#00D4AA', width=3),
        marker=dict(color=[colors_full[0]], size=10, line=dict(color='white', width=2)),
        name='Journey',
        text=[emoji_labels[0]],
        hovertemplate='<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>',
        customdata=[plot_data['confidence'].iloc[0] * 100],
    ))
    
    # Build animation frames
    frames = []
    for i in range(1, n_frames + 1):
        frame_data = plot_data.iloc[:i]
        frame_colors = colors_full[:i]
        frame_emojis = emoji_labels[:i]
        
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=list(range(n_frames)),  # Full path (faint)
                    y=plot_data['emotion_num'],
                    mode='lines',
                    line=dict(color='#30363D', width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate='<extra></extra>',
                ),
                go.Scatter(
                    x=list(range(i)),
                    y=frame_data['emotion_num'],
                    mode='lines+markers',
                    line=dict(color='#00D4AA', width=3),
                    marker=dict(color=frame_colors, size=8,
                                line=dict(color='white', width=1.5)),
                    name='Journey',
                    text=frame_emojis,
                    hovertemplate='<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>',
                    customdata=frame_data['confidence'].values * 100,
                ),
            ],
            name=f'frame{i}',
            traces=[0, 1],
        ))
    
    fig.frames = frames
    
    # Duration per frame (adjusted by speed)
    base_duration = 800  # ms per frame at 1x
    frame_duration = max(50, int(base_duration / speed_mult))
    
    # Animation settings
    updatemenus = [{
        'type': 'buttons',
        'showactive': False,
        'x': 0.5,
        'y': -0.15,
        'xanchor': 'center',
        'buttons': [
            {
                'label': '▶️ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': frame_duration, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0},
                }],
            },
            {
                'label': '⏸️ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0},
                }],
            },
        ],
    }]
    
    # Slider for manual frame navigation
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 14, 'color': '#E6EDF3'},
            'prefix': 'Frame: ',
            'visible': True,
            'xanchor': 'right',
        },
        'transition': {'duration': 50},
        'pad': {'b': 10},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {
                'args': [[f'frame{k+1}'], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0},
                }],
                'label': str(k + 1),
                'method': 'animate',
            }
            for k in range(n_frames)
        ],
    }]
    
    fig.update_layout(
        title=dict(
            text=f"🎬 Emotion Journey — {recording['name']}",
            font=dict(size=14, color='#E6EDF3'),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=[f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            gridcolor='#30363D',
            range=[-0.5, 6.5],
        ),
        xaxis=dict(
            title="Frame (progression)",
            gridcolor='#30363D',
            range=[-0.5, n_frames - 0.5],
        ),
        height=500,
        updatemenus=updatemenus,
        sliders=sliders,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=100),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ─── Emotion Distribution for This Recording ─────────────
    st.markdown("---")
    st.markdown("### 📊 Recording Breakdown")
    
    dist = df_rec['emotion'].value_counts()
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in dist.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in dist.index]
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=emoji_labels,
            values=dist.values,
            marker=dict(colors=colors, line=dict(color='#161B22', width=2)),
            textinfo='label+percent',
            textfont=dict(color='#E6EDF3', size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        )])
        fig_pie.update_layout(
            title="Emotion Distribution",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_d2:
        # Confidence over time for this recording
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=list(range(len(df_rec))),
            y=df_rec['confidence'] * 100,
            mode='lines',
            line=dict(color='#00D4AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)',
            name='Confidence',
            hovertemplate='Frame %{x}<br>Confidence: %{y:.1f}%<extra></extra>',
        ))
        fig_conf.update_layout(
            title="Confidence Over Time",
            yaxis=dict(title="Confidence (%)", range=[0, 100], gridcolor='#30363D'),
            xaxis=dict(title="Frame", gridcolor='#30363D'),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # ─── Delete Recording Button ─────────────────────────────
    if st.button("🗑️ Delete This Recording", type="secondary", use_container_width=True):
        idx = st.session_state.timeline_recordings.index(recording)
        st.session_state.timeline_recordings.pop(idx)
        st.rerun()