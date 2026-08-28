"""Tests for EmotionSmoother class."""

import pytest

from utils.smoothing_utils import EmotionSmoother


class TestEmotionSmoother:
    """Tests for EmotionSmoother."""

    def test_init(self):
        smoother = EmotionSmoother()
        assert smoother.window == 5

    def test_single_update(self):
        smoother = EmotionSmoother()
        probs = [0.1, 0.0, 0.0, 0.8, 0.05, 0.05, 0.0]
        smoothed = smoother.update(probs)
        assert len(smoothed) == 7

    def test_smoothed_emotion(self):
        smoother = EmotionSmoother()
        probs = [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]
        emotion, conf, smoothed = smoother.smoothed_emotion(probs)
        assert emotion == "Happy"
        assert conf > 0

    def test_reset(self):
        smoother = EmotionSmoother()
        smoother.update([0.0] * 7)
        smoother.update([0.0] * 7)
        smoother.reset()
        assert len(smoother._buffer) == 0

    def test_window_limit(self):
        smoother = EmotionSmoother(window=3)
        for _ in range(10):
            smoother.update([0.1, 0.0, 0.0, 0.8, 0.05, 0.05, 0.0])
        assert len(smoother._buffer) <= 3

    def test_averaging(self):
        smoother = EmotionSmoother(window=3)
        # Mix of Happy and Sad
        smoother.update([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Happy
        smoother.update([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Sad
        smoother.update([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Happy
        smoothed = smoother.update([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        # Should average all 4
        assert smoothed[3] > 0  # Happy component
        assert smoothed[5] > 0  # Sad component
