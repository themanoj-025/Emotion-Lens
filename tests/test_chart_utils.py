"""Tests for EmotionLens chart_utils module — Plotly chart builders."""

import plotly.graph_objects as go

from utils.chart_utils import (
    emotion_bar_chart,
    emotion_pie,
    emotion_radar_chart,
    emotion_timeline,
    valence_arousal_scatter,
)
from utils.config import EMOTIONS


def test_bar_chart_returns_figure():
    probs = [0.1, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    fig = emotion_bar_chart(probs)
    assert isinstance(fig, go.Figure)


def test_bar_chart_has_seven_traces():
    probs = [0.1, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    fig = emotion_bar_chart(probs)
    assert len(fig.data) == 7


def test_bar_chart_uses_emotion_names():
    probs = [0.1, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    fig = emotion_bar_chart(probs)
    y_labels = [trace.y[0] for trace in fig.data]
    assert y_labels == EMOTIONS


def test_radar_chart_returns_figure():
    probs = [0.1, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    fig = emotion_radar_chart(probs)
    assert isinstance(fig, go.Figure)


def test_radar_chart_closes_loop():
    """Radar chart should have probs[0] appended to close the polygon."""
    probs = [0.1, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    fig = emotion_radar_chart(probs)
    trace = fig.data[0]
    assert len(trace.r) == len(probs) + 1
    assert trace.r[-1] == probs[0]


def test_timeline_empty_returns_figure():
    fig = emotion_timeline([])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_timeline_has_seven_traces():
    predictions = [
        {"emotion": "Happy", "confidence": 0.9, "probabilities": [0.05] * 7, "timestamp": "2026-01-01T00:00:00"},
        {"emotion": "Sad", "confidence": 0.8, "probabilities": [0.05] * 7, "timestamp": "2026-01-01T00:00:01"},
    ]
    fig = emotion_timeline(predictions)
    assert len(fig.data) == 7


def test_pie_empty_returns_figure():
    fig = emotion_pie([])
    assert isinstance(fig, go.Figure)


def test_pie_groups_emotions():
    predictions = [
        {"emotion": "Happy", "confidence": 0.9, "probabilities": [0] * 7, "timestamp": "t1"},
        {"emotion": "Happy", "confidence": 0.8, "probabilities": [0] * 7, "timestamp": "t2"},
        {"emotion": "Sad", "confidence": 0.7, "probabilities": [0] * 7, "timestamp": "t3"},
    ]
    fig = emotion_pie(predictions)
    assert len(fig.data) == 1  # single Pie trace
    labels = fig.data[0].labels
    assert "Happy" in labels
    assert "Sad" in labels


def test_valence_arousal_empty():
    fig = valence_arousal_scatter([])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_valence_arousal_has_points():
    predictions = [
        {"emotion": "Happy", "confidence": 0.9, "probabilities": [0] * 7, "timestamp": "t1"},
        {"emotion": "Angry", "confidence": 0.8, "probabilities": [0] * 7, "timestamp": "t2"},
    ]
    fig = valence_arousal_scatter(predictions)
    assert len(fig.data) == 1  # single Scatter trace
    assert len(fig.data[0].x) == 2
