"""Tests for EmotionLens 🎭 temporal smoothing — reduces live-detection flickering."""

import numpy as np
import pytest

from utils.smoothing_utils import EmotionSmoother


def make_probs(emotion_idx, confidence=0.9) -> None:
    """Deterministic one-hot-ish probability vector of length 7."""
    probs = [0.05] * 7
    probs[emotion_idx] = confidence
    return probs


def test_init_default_window() -> None:
    s = EmotionSmoother()
    assert s.window == 5


def test_single_update_returns_same() -> None:
    s = EmotionSmoother(window=5)
    probs = make_probs(3)  # Happy
    out = s.update(probs)
    assert len(out) == 7
    # With one sample the mean equals the sample
    assert np.allclose(out, probs)


def test_smoothing_averages_values() -> None:
    s = EmotionSmoother(window=3)
    s.update([1.0, 0.0])
    s.update([1.0, 0.0])
    out = s.update([0.0, 1.0])
    assert np.allclose(out, [2.0 / 3.0, 1.0 / 3.0])


def test_window_bounds() -> None:
    s = EmotionSmoother(window=2)
    s.update([1.0, 0.0])
    s.update([1.0, 0.0])
    s.update([0.0, 1.0])
    out = s.update([0.0, 1.0])
    # Buffer limited to 2 → mean of last two
    assert np.allclose(out, [0.0, 1.0])


def test_smoothed_emotion_returns_tuple() -> None:
    s = EmotionSmoother(window=5)
    for _ in range(3):
        s.update(make_probs(4))  # Neutral
    emotion, confidence, probs = s.smoothed_emotion(make_probs(4))
    assert emotion == "Neutral"
    assert 0.0 <= confidence <= 1.0
    assert len(probs) == 7


def test_smoothed_emotion_dominates() -> None:
    """3 Neutral frames followed by a Happy outlier → Neutral still wins."""
    s = EmotionSmoother(window=5)
    for _ in range(3):
        s.update(make_probs(4))
    emotion, _, _ = s.smoothed_emotion(make_probs(3))
    assert emotion == "Neutral"


def test_reset_clears_buffer() -> None:
    s = EmotionSmoother(window=3)
    s.update(make_probs(3))
    s.update(make_probs(3))
    s.reset()
    out = s.update(make_probs(3))
    assert np.allclose(out, make_probs(3))


def test_probabilities_normalize_shape() -> None:
    """Update accepts any length and preserves it (app-level asserts 7)."""
    s = EmotionSmoother()
    out = s.update([0.1, 0.2, 0.7])
    assert len(out) == 3


@pytest.mark.parametrize("window", [1, 2, 10])
def test_custom_windows(window) -> None:
    s = EmotionSmoother(window=window)
    for _ in range(12):
        out = s.update(make_probs(3))
        assert len(out) == 7
