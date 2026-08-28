"""Chart renderers for analytics dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.emotion_utils import compute_positivity_score, render_mood_music_card
from utils.model_utils import EMOTION_CONFIG, EMOTIONS, PLOTLY_THEME


def _render_distribution_chart(df) -> None:
    """Render emotion distribution pie and bar charts."""
    st.markdown("#### Emotion Distribution")

    dist = df["emotion"].value_counts()
    colors = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in dist.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in dist.index]

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=emoji_labels,
                    values=dist.values,
                    marker={"colors": colors, "line": {"color": "#161B22", "width": 2}},
                    textinfo="label+percent",
                    textfont={"color": "#E6EDF3", "size": 12},
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                )
            ]
        )
        fig_pie.update_layout(
            height=400,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=emoji_labels,
                    y=dist.values,
                    marker_color=colors,
                    text=dist.values,
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                )
            ]
        )
        fig_bar.update_layout(
            title="Count per Emotion",
            yaxis={"title": "Count", "gridcolor": "#30363D"},
            height=400,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_timeline_chart(df) -> None:
    """Render emotion over time line chart."""
    st.markdown("#### Emotion Timeline")

    if "timestamp" not in df.columns:
        st.warning("Timestamp data not available.")
        return

    df_time = df.copy()
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
    df_time = df_time.sort_values("timestamp")

    # Numeric encoding for emotions
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    df_time["emotion_num"] = df_time["emotion"].map(emotion_to_num)

    # Create scatter trace with color-coded markers
    colors = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in df_time["emotion"]]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in df_time["emotion"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_time["timestamp"],
            y=df_time["emotion_num"],
            mode="lines+markers",
            line={"color": "#00D4AA", "width": 2},
            marker={
                "color": colors,
                "size": 8,
                "line": {"color": "#161B22", "width": 1},
            },
            text=emoji_labels,
            hovertemplate="<b>%{text}</b><br>Time: %{x|%H:%M:%S}<br>Confidence: %{customdata:.1f}%<extra></extra>",
            customdata=df_time["confidence"] * 100,
        )
    )

    fig.update_layout(
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(7)),
            "ticktext": [f"{EMOTION_CONFIG[e]['emoji']} {e}" for e in EMOTIONS],
            "gridcolor": "#30363D",
            "title": "Emotion",
        },
        xaxis={
            "title": "Time",
            "gridcolor": "#30363D",
            "tickformat": "%H:%M:%S",
        },
        height=400,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 30, "b": 30},
        **PLOTLY_THEME,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_chart(df) -> None:
    """Render average confidence per emotion bar chart."""
    st.markdown("#### Average Confidence per Emotion")

    avg_conf = df.groupby("emotion")["confidence"].mean() * 100
    avg_conf = avg_conf.reindex(EMOTIONS, fill_value=0)

    colors = [EMOTION_CONFIG.get(e, {}).get("color", "#95A5A6") for e in avg_conf.index]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in avg_conf.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=emoji_labels,
                y=avg_conf.values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in avg_conf.values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg Confidence: %{y:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        yaxis={"title": "Average Confidence (%)", "range": [0, 100], "gridcolor": "#30363D"},
        height=400,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        **PLOTLY_THEME,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(df) -> None:
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
        curr = recent.iloc[i]["emotion"]
        next_e = recent.iloc[i + 1]["emotion"]
        if curr in emotion_order and next_e in emotion_order:
            transition_matrix[emotion_order.index(curr)][emotion_order.index(next_e)] += 1

    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition_matrix = transition_matrix / row_sums

    labels = [f"{EMOTION_CONFIG[e]['emoji']} {e[:3]}" for e in emotion_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=transition_matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=np.round(transition_matrix, 2),
            texttemplate="%{text:.0%}",
            textfont={"size": 10, "color": "#E6EDF3"},
            hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Emotion Transition Probabilities",
        xaxis={"title": "Next Emotion", "tickfont": {"size": 10}},
        yaxis={"title": "Current Emotion", "tickfont": {"size": 10}},
        height=450,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        **PLOTLY_THEME,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_positivity_analysis(df) -> None:
    """Render positivity/valence score trends."""
    st.markdown("#### Positivity Score Over Time")

    if len(df) < 2:
        st.info("Need more data to show positivity trend.")
        return

    # Compute positivity for each prediction
    positivity_scores = []
    for _, row in df.iterrows():
        probs = row.get("probabilities")
        if probs and len(probs) == 7:
            score = compute_positivity_score(probs)
            positivity_scores.append(score)
        else:
            positivity_scores.append(0)

    df.tail(100).copy()
    scores = positivity_scores[-100:]

    # Color mapping
    colors = ["#FF6B6B" if s < -0.3 else "#F5A623" if s < 0.3 else "#2ECC71" for s in scores]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            mode="lines+markers",
            name="Positivity Score",
            line={"color": "#00D4AA", "width": 2},
            marker={"color": colors, "size": 5},
            hovertemplate="Frame %{x}<br>Score: %{y:+.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#8B949E", opacity=0.5)

    # Add rolling average
    if len(scores) > 5:
        window = min(10, len(scores) // 2)
        rolling_avg = pd.Series(scores).rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=rolling_avg,
                mode="lines",
                name=f"{window}-frame Average",
                line={"color": "#FFFFFF", "width": 2, "dash": "dot"},
                hovertemplate="Frame %{x}<br>Avg Score: %{y:+.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_layout(
        yaxis={
            "title": "Positivity Score",
            "range": [-1.1, 1.1],
            "gridcolor": "#30363D",
            "tickvals": [-1, -0.5, 0, 0.5, 1],
            "ticktext": ["-1.0 😟", "-0.5", "0 😐", "0.5", "1.0 😊"],
        },
        xaxis={"title": "Frame (recent)", "gridcolor": "#30363D"},
        height=400,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 30, "b": 30},
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
    top_emotion = df["emotion"].mode().iloc[0] if not df.empty else None
    if top_emotion:
        render_mood_music_card(top_emotion)


