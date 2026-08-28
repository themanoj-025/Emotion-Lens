"""Timeline replay feature for analytics dashboard."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.emotion_utils import compute_positivity_score
from utils.model_utils import EMOTION_CONFIG, EMOTIONS, PLOTLY_THEME


def _render_timeline_replay(df) -> None:
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

    # Controls
    rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 1])

    currently_recording = st.session_state.timeline_recording

    with rec_col1:
        record_duration = st.slider(
            "Recording Duration (seconds)",
            min_value=10,
            max_value=120,
            value=60,
            step=10,
            disabled=currently_recording,
            help="How long to record predictions. The dashboard will collect all predictions made during this window.",
        )

    with rec_col2:
        if not currently_recording:
            if st.button(
                "🔴 Start Recording",
                type="primary",
                use_container_width=True,
                disabled=df.empty,
            ):
                # Start recording: mark the current prediction count as the start
                st.session_state.timeline_recording = True
                st.session_state.timeline_start_idx = len(st.session_state.get("predictions", []))
                st.session_state.timeline_recording_start = datetime.now()
                st.session_state.timeline_recording_end = record_duration
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                _save_timeline_recording()
                st.rerun()

    with rec_col3:
        if st.session_state.timeline_recordings:
            recording_names = [r["name"] for r in st.session_state.timeline_recordings]
            selected_recording = st.selectbox(
                "Previous Recordings",
                options=list(range(len(recording_names))),
                format_func=lambda i: recording_names[i],
                index=len(recording_names) - 1,
                help="Select a previous recording to replay.",
            )
        else:
            selected_recording = None
            st.markdown(
                "<p style='color:#8B949E; padding-top:1.8rem;'>No recordings yet</p>",
                unsafe_allow_html=True,
            )

    # Recording Progress
    if currently_recording:
        elapsed = (datetime.now() - st.session_state.timeline_recording_start).total_seconds()
        remaining = max(0, st.session_state.timeline_recording_end - elapsed)
        progress = min(1.0, elapsed / st.session_state.timeline_recording_end)

        st.progress(progress)

        # Show live recording stats
        current_count = len(st.session_state.get("predictions", []))
        new_preds = current_count - st.session_state.timeline_start_idx

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("⏱️ Time Remaining", f"{remaining:.0f}s")
        col_b.metric("📊 Predictions Recorded", new_preds)
        col_c.metric("📈 Progress", f"{progress * 100:.0f}%")

        # Auto-stop when duration reached
        if elapsed >= st.session_state.timeline_recording_end:
            _save_timeline_recording()
            st.rerun()

        st.info(
            f"🔴 Recording in progress... Auto-stop in {remaining:.0f} seconds. Make predictions using the Live Camera or Image Analysis pages."
        )

        # Show a placeholder chart with what's been recorded so far
        current_preds = st.session_state.get("predictions", [])
        new_slice = current_preds[st.session_state.timeline_start_idx :]
        if len(new_slice) >= 2:
            _render_live_recording_chart(new_slice)

        return  # Don't show playback while recording

    # Playback Mode
    if selected_recording is not None and st.session_state.timeline_recordings:
        recording = st.session_state.timeline_recordings[selected_recording]
        _render_playback_view(recording)
    else:
        if df.empty:
            st.info(
                "📭 No predictions yet. Use the **Live Camera** or **Image Analysis** pages to collect data first."
            )
        else:
            st.info(
                "👆 Press **🔴 Start Recording** to begin capturing predictions. The replay will appear here once recording completes."
            )


def _save_timeline_recording() -> None:
    """Save the current recording to session state."""
    predictions = st.session_state.get("predictions", [])
    start_idx = st.session_state.timeline_start_idx
    recording_slice = predictions[start_idx:]

    if not recording_slice:
        st.session_state.timeline_recording = False
        return

    start_time = st.session_state.get("timeline_recording_start", datetime.now())
    duration = st.session_state.get("timeline_recording_end", 60)

    timestamp_str = start_time.strftime("%H:%M:%S")
    emotion_counts = {}
    for p in recording_slice:
        e = p["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    top_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "—"
    emoji = EMOTION_CONFIG.get(top_emotion, {}).get("emoji", "")

    st.session_state.timeline_recordings.append(
        {
            "name": f"{timestamp_str} — {emoji} {top_emotion} ({len(recording_slice)} preds)",
            "predictions": recording_slice,
            "start_time": start_time.isoformat(),
            "duration": duration,
            "count": len(recording_slice),
            "top_emotion": top_emotion,
        }
    )
    st.session_state.timeline_recording = False


def _render_live_recording_chart(predictions_slice) -> None:
    """Render a live preview of the current recording."""
    df_rec = pd.DataFrame(predictions_slice)
    if "timestamp" in df_rec.columns:
        df_rec["timestamp"] = pd.to_datetime(df_rec["timestamp"])
        df_rec = df_rec.sort_values("timestamp")

    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    df_rec["emotion_num"] = df_rec["emotion"].map(emotion_to_num)
    colors = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in df_rec["emotion"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df_rec))),
            y=df_rec["emotion_num"],
            mode="lines+markers",
            line={"color": "#00D4AA", "width": 2},
            marker={"color": colors, "size": 6},
            text=[f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in df_rec["emotion"]],
            hovertemplate="<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>",
            customdata=df_rec["confidence"] * 100,
        )
    )
    fig.update_layout(
        title="🔴 Live Recording Preview",
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(7)),
            "ticktext": [f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            "gridcolor": "#30363D",
        },
        xaxis={"title": "Frame (since recording started)", "gridcolor": "#30363D"},
        height=300,
        margin={"l": 20, "r": 20, "t": 40, "b": 30},
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_playback_view(recording) -> None:
    """Render the animated playback view for a completed recording."""
    preds = recording["predictions"]
    count = recording["count"]

    df_rec = pd.DataFrame(preds)
    if "timestamp" in df_rec.columns:
        df_rec["timestamp"] = pd.to_datetime(df_rec["timestamp"])
        df_rec = df_rec.sort_values("timestamp")

    # Recording Summary
    st.markdown("---")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("📊 Predictions", count)
    with col_s2:
        st.metric("⏱️ Duration", f"{recording['duration']}s")
    with col_s3:
        top_emoji = EMOTION_CONFIG.get(recording["top_emotion"], {}).get("emoji", "")
        st.metric("🎯 Dominant", f"{top_emoji} {recording['top_emotion']}")
    with col_s4:
        # Compute average positivity
        positivity_scores = []
        for _, row in df_rec.iterrows():
            probs = row.get("probabilities")
            if probs and len(probs) == 7:
                positivity_scores.append(compute_positivity_score(probs))
        avg_pos = np.mean(positivity_scores) if positivity_scores else 0
        mood = "😊" if avg_pos > 0.3 else "😟" if avg_pos < -0.3 else "😐"
        st.metric("💚 Mood", f"{mood} {avg_pos:+.2f}")

    # Playback Speed Control
    st.markdown("---")
    play_col1, play_col2 = st.columns([1, 3])

    with play_col1:
        playback_speed = st.select_slider(
            "Playback Speed",
            options=["0.25x", "0.5x", "1x", "2x", "4x"],
            value="1x",
            help="Controls how fast the animation plays. 1x = real-time speed.",
        )
        speed_mult = float(playback_speed.replace("x", ""))

    with play_col2:
        st.markdown("<p style='padding-top:0.5rem;'></p>", unsafe_allow_html=True)
        play_btn_col, restart_col = st.columns(2)
        with play_btn_col:
            st.button("▶️ Play Animation", type="primary", use_container_width=True)
        with restart_col:
            st.button("⏹️ Reset", use_container_width=True)

    # Animated Chart with Frames
    if count < 2:
        st.info("Need at least 2 data points for playback.")
        return

    # Compute display data
    n_points = min(count, 300)  # Cap at 300 for performance
    step = max(1, count // n_points)
    plot_data = df_rec.iloc[::step].head(n_points).reset_index(drop=True)
    n_frames = len(plot_data)

    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    plot_data["emotion_num"] = plot_data["emotion"].map(emotion_to_num)

    # Create the base figure (empty initially)
    fig = go.Figure()

    # Add the full trace as a faint background reference
    fig.add_trace(
        go.Scatter(
            x=list(range(n_frames)),
            y=plot_data["emotion_num"],
            mode="lines",
            line={"color": "#30363D", "width": 1, "dash": "dot"},
            name="Full path",
            hovertemplate="<extra></extra>",
            showlegend=False,
        )
    )

    # Add the animated trace (starts empty, filled by frames)
    colors_full = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in plot_data["emotion"]]
    emoji_labels = [
        f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in plot_data["emotion"]
    ]

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[plot_data["emotion_num"].iloc[0]],
            mode="lines+markers",
            line={"color": "#00D4AA", "width": 3},
            marker={"color": [colors_full[0]], "size": 10, "line": {"color": "white", "width": 2}},
            name="Journey",
            text=[emoji_labels[0]],
            hovertemplate="<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>",
            customdata=[plot_data["confidence"].iloc[0] * 100],
        )
    )

    # Build animation frames
    frames = []
    for i in range(1, n_frames + 1):
        frame_data = plot_data.iloc[:i]
        frame_colors = colors_full[:i]
        frame_emojis = emoji_labels[:i]

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=list(range(n_frames)),  # Full path (faint)
                        y=plot_data["emotion_num"],
                        mode="lines",
                        line={"color": "#30363D", "width": 1, "dash": "dot"},
                        showlegend=False,
                        hovertemplate="<extra></extra>",
                    ),
                    go.Scatter(
                        x=list(range(i)),
                        y=frame_data["emotion_num"],
                        mode="lines+markers",
                        line={"color": "#00D4AA", "width": 3},
                        marker={
                            "color": frame_colors,
                            "size": 8,
                            "line": {"color": "white", "width": 1.5},
                        },
                        name="Journey",
                        text=frame_emojis,
                        hovertemplate="<b>%{text}</b><br>Frame: %{x}<br>Confidence: %{customdata:.1f}%<extra></extra>",
                        customdata=frame_data["confidence"].values * 100,
                    ),
                ],
                name=f"frame{i}",
                traces=[0, 1],
            )
        )

    fig.frames = frames

    # Duration per frame (adjusted by speed)
    base_duration = 800  # ms per frame at 1x
    frame_duration = max(50, int(base_duration / speed_mult))

    # Animation settings
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.5,
            "y": -0.15,
            "xanchor": "center",
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    # Slider for manual frame navigation
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": "#E6EDF3"},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 50},
            "pad": {"b": 10},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f"frame{k + 1}"],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": str(k + 1),
                    "method": "animate",
                }
                for k in range(n_frames)
            ],
        }
    ]

    fig.update_layout(
        title={
            "text": f"🎬 Emotion Journey — {recording['name']}",
            "font": {"size": 14, "color": "#E6EDF3"},
        },
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(7)),
            "ticktext": [f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            "gridcolor": "#30363D",
            "range": [-0.5, 6.5],
        },
        xaxis={
            "title": "Frame (progression)",
            "gridcolor": "#30363D",
            "range": [-0.5, n_frames - 0.5],
        },
        height=500,
        updatemenus=updatemenus,
        sliders=sliders,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 100},
        **PLOTLY_THEME,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Emotion Distribution for This Recording
    st.markdown("---")
    st.markdown("### 📊 Recording Breakdown")

    dist = df_rec["emotion"].value_counts()
    colors = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in dist.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in dist.index]

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=emoji_labels,
                    values=dist.values,
                    marker={"colors": colors, "line": {"color": "#161B22", "width": 2}},
                    textinfo="label+percent",
                    textfont={"color": "#E6EDF3", "size": 11},
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                )
            ]
        )
        fig_pie.update_layout(
            title="Emotion Distribution",
            height=350,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_d2:
        # Confidence over time for this recording
        fig_conf = go.Figure()
        fig_conf.add_trace(
            go.Scatter(
                x=list(range(len(df_rec))),
                y=df_rec["confidence"] * 100,
                mode="lines",
                line={"color": "#00D4AA", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0, 212, 170, 0.1)",
                name="Confidence",
                hovertemplate="Frame %{x}<br>Confidence: %{y:.1f}%<extra></extra>",
            )
        )
        fig_conf.update_layout(
            title="Confidence Over Time",
            yaxis={"title": "Confidence (%)", "range": [0, 100], "gridcolor": "#30363D"},
            xaxis={"title": "Frame", "gridcolor": "#30363D"},
            height=350,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # Delete Recording Button
    if st.button("🗑️ Delete This Recording", type="secondary", use_container_width=True):
        idx = st.session_state.timeline_recordings.index(recording)
        st.session_state.timeline_recordings.pop(idx)
        st.rerun()
