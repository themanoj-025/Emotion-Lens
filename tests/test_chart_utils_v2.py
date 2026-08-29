"""Tests for Plotly chart builders."""


from utils.chart_utils import (
    emotion_bar_chart,
    emotion_pie,
    emotion_radar_chart,
    emotion_timeline,
    valence_arousal_scatter,
)


class TestEmotionBarChart:
    """Tests for emotion_bar_chart."""

    def test_creates_figure(self):
        probs = [0.1, 0.0, 0.0, 0.8, 0.05, 0.05, 0.0]
        fig = emotion_bar_chart(probs)
        assert fig is not None
        assert len(fig.data) == 7

    def test_custom_title(self):
        probs = [0.0] * 7
        fig = emotion_bar_chart(probs, title="Custom")
        assert fig.layout.title.text == "Custom"


class TestEmotionRadarChart:
    """Tests for emotion_radar_chart."""

    def test_creates_figure(self):
        probs = [0.1, 0.0, 0.0, 0.8, 0.05, 0.05, 0.0]
        fig = emotion_radar_chart(probs)
        assert fig is not None
        assert len(fig.data) == 1


class TestEmotionTimeline:
    """Tests for emotion_timeline."""

    def test_empty_predictions(self):
        fig = emotion_timeline([])
        assert fig is not None

    def test_with_predictions(self):
        preds = [
            {"emotion": "Happy", "confidence": 0.9, "probabilities": [0]*7, "timestamp": "2025-01-01"},
            {"emotion": "Sad", "confidence": 0.7, "probabilities": [0]*7, "timestamp": "2025-01-02"},
        ]
        fig = emotion_timeline(preds)
        assert fig is not None


class TestEmotionPie:
    """Tests for emotion_pie."""

    def test_empty_predictions(self):
        fig = emotion_pie([])
        assert fig is not None

    def test_with_predictions(self):
        preds = [
            {"emotion": "Happy", "confidence": 0.9},
            {"emotion": "Happy", "confidence": 0.8},
            {"emotion": "Sad", "confidence": 0.7},
        ]
        fig = emotion_pie(preds)
        assert fig is not None
        assert len(fig.data) == 1  # Single pie trace


class TestValenceArousalScatter:
    """Tests for valence_arousal_scatter."""

    def test_empty_predictions(self):
        fig = valence_arousal_scatter([])
        assert fig is not None

    def test_with_predictions(self):
        preds = [
            {"emotion": "Happy", "confidence": 0.9},
            {"emotion": "Sad", "confidence": 0.7},
        ]
        fig = valence_arousal_scatter(preds)
        assert fig is not None
