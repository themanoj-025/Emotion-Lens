"""Reusable Plotly chart builders with the global dark theme."""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter
from utils.config import EMOTIONS, EMOTION_CONFIG

LAYOUT_BASE = dict(
    paper_bgcolor='#161B22', plot_bgcolor='#0D1117',
    font=dict(color='#E6EDF3', family='Space Grotesk, sans-serif', size=12),
    margin=dict(l=16, r=16, t=32, b=16),
)


def emotion_bar_chart(probs: list[float], title: str = "Emotion Probabilities") -> go.Figure:
    """Horizontal bar chart — 7 emotions with their individual colors."""
    fig = go.Figure()
    for emotion, prob in zip(EMOTIONS, probs):
        fig.add_trace(go.Bar(
            x=[prob], y=[emotion],
            orientation='h',
            marker_color=EMOTION_CONFIG[emotion]['color'],
            marker_line_width=0,
            showlegend=False,
            hovertemplate=f"<b>{emotion}</b>: {prob*100:.1f}%<extra></extra>",
        ))
    fig.update_layout(**LAYOUT_BASE, title=title, xaxis_tickformat='.0%',
                      xaxis_range=[0, 1], height=260, yaxis_categoryorder='array',
                      yaxis_categoryarray=EMOTIONS[::-1])
    return fig


def emotion_radar_chart(probs: list[float], title: str = "Emotion Profile") -> go.Figure:
    """Radar/Spider chart for emotion probabilities."""
    fig = go.Figure(go.Scatterpolar(
        r=probs + [probs[0]],
        theta=EMOTIONS + [EMOTIONS[0]],
        fill='toself',
        fillcolor='rgba(0,212,170,0.15)',
        line=dict(color='#00D4AA', width=2),
    ))
    fig.update_layout(**LAYOUT_BASE, polar=dict(
        bgcolor='#161B22',
        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(139,148,158,0.15)', color='#8B949E'),
        angularaxis=dict(gridcolor='rgba(139,148,158,0.15)', color='#E6EDF3'),
    ), title=title, height=320)
    return fig


def emotion_timeline(predictions: list[dict]) -> go.Figure:
    """Line chart of emotion confidence over time (from session history)."""
    if not predictions:
        return go.Figure()
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig = go.Figure()
    for emotion in EMOTIONS:
        ys = [p['probabilities'][EMOTIONS.index(emotion)] for p in predictions]
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=ys,
            name=emotion,
            line=dict(color=EMOTION_CONFIG[emotion]['color'], width=1.5),
            hovertemplate=f"<b>{emotion}</b>: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(**LAYOUT_BASE, title="Emotion Timeline", height=300,
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
                      xaxis_title=None, yaxis_tickformat='.0%')
    return fig


def emotion_pie(predictions: list[dict]) -> go.Figure:
    """Donut chart — distribution of detected emotions."""
    if not predictions:
        return go.Figure()
    counts = Counter(p['emotion'] for p in predictions)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [EMOTION_CONFIG[e]['color'] for e in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#0D1117', width=2)),
        hole=0.55,
        hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Session Emotion Distribution",
                      height=300, showlegend=True)
    return fig


def valence_arousal_scatter(predictions: list[dict]) -> go.Figure:
    """2D valence-arousal plot (Russell circumplex model)."""
    if not predictions:
        return go.Figure()
    valences = [EMOTION_CONFIG[p['emotion']]['valence'] for p in predictions]
    arousals = [EMOTION_CONFIG[p['emotion']]['arousal'] for p in predictions]
    colors = [EMOTION_CONFIG[p['emotion']]['color'] for p in predictions]
    emotions = [p['emotion'] for p in predictions]

    fig = go.Figure(go.Scatter(
        x=valences, y=arousals, mode='markers',
        marker=dict(color=colors, size=8, opacity=0.7, line=dict(color='#0D1117', width=1)),
        text=emotions,
        hovertemplate="<b>%{text}</b><br>Valence: %{x:.1f}, Arousal: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color='rgba(139,148,158,0.3)', dash='dot'))
    fig.add_vline(x=0, line=dict(color='rgba(139,148,158,0.3)', dash='dot'))
    fig.update_layout(**LAYOUT_BASE, title="Valence–Arousal Map",
                      xaxis_title="Valence (negative ← → positive)",
                      yaxis_title="Arousal (calm ← → excited)",
                      xaxis_range=[-1.2, 1.2], yaxis_range=[-1.2, 1.2], height=360)
    return fig
