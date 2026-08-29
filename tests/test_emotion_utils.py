"""Tests for Emotion-Lens: config helpers and emotion_utils pure functions."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock tensorflow before importing emotion_utils
sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())
sys.modules.setdefault("tensorflow.keras.preprocessing", MagicMock())
sys.modules.setdefault("tensorflow.keras.preprocessing.image", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from utils.config import (
    EMOTIONS,
    EMOTION_CONFIG,
    SMOOTHING_WINDOW,
    MAX_HISTORY,
    GAME_COUNTDOWN,
    positivity_score,
    emotion_index,
    BADGES,
)


# ── Config Constants ────────────────────────────────────────────────────────


class TestConfig:
    """Tests for config constants and helper functions."""

    def test_emotions_has_7_entries(self) -> None:
        assert len(EMOTIONS) == 7

    def test_emotions_order(self) -> None:
        assert EMOTIONS == ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def test_all_emotions_have_config(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG

    def test_emotion_config_has_required_keys(self) -> None:
        for emotion in EMOTION_CONFIG:
            cfg = EMOTION_CONFIG[emotion]
            assert "color" in cfg
            assert "bg" in cfg
            assert "emoji" in cfg
            assert "valence" in cfg
            assert "arousal" in cfg

    def test_happy_is_positive_valence(self) -> None:
        assert EMOTION_CONFIG["Happy"]["valence"] == 1.0

    def test_angry_is_negative_valence(self) -> None:
        assert EMOTION_CONFIG["Angry"]["valence"] == -1.0

    def test_neutral_zero_arousal(self) -> None:
        assert EMOTION_CONFIG["Neutral"]["arousal"] == 0.0

    def test_smoothing_window(self) -> None:
        assert SMOOTHING_WINDOW == 5

    def test_max_history(self) -> None:
        assert MAX_HISTORY == 300

    def test_game_countdown(self) -> None:
        assert GAME_COUNTDOWN == 10


class TestPositivityScore:
    """Tests for the positivity scoring function."""

    def test_all_happy(self) -> None:
        probs = [0, 0, 0, 1, 0, 0, 0]  # Happy = 1.0
        score = positivity_score(probs)
        assert abs(score - 1.0) < 1e-6

    def test_all_angry(self) -> None:
        probs = [1, 0, 0, 0, 0, 0, 0]  # Angry = -1.0
        score = positivity_score(probs)
        assert abs(score - (-1.0)) < 1e-6

    def test_all_neutral(self) -> None:
        probs = [0, 0, 0, 0, 1, 0, 0]  # Neutral = 0.0
        score = positivity_score(probs)
        assert abs(score) < 1e-6

    def test_equal_distribution(self) -> None:
        probs = [1 / 7] * 7
        score = positivity_score(probs)
        assert -1.0 <= score <= 1.0

    def test_mixed_positive(self) -> None:
        probs = [0.1, 0.0, 0.0, 0.6, 0.2, 0.0, 0.1]  # Happy dominant
        score = positivity_score(probs)
        assert score > 0

    def test_mixed_negative(self) -> None:
        probs = [0.4, 0.1, 0.1, 0.0, 0.0, 0.3, 0.1]  # Angry+Sad dominant
        score = positivity_score(probs)
        assert score < 0


class TestEmotionIndex:
    """Tests for emotion index lookup."""

    def test_happy_index(self) -> None:
        assert emotion_index("Happy") == 3

    def test_angry_index(self) -> None:
        assert emotion_index("Angry") == 0

    def test_surprise_index(self) -> None:
        assert emotion_index("Surprise") == 6

    def test_neutral_index(self) -> None:
        assert emotion_index("Neutral") == 4


class TestBadges:
    """Tests for badge evaluation logic."""

    def test_badges_count(self) -> None:
        assert len(BADGES) == 7

    def test_smile_master_badge(self) -> None:
        check = BADGES["😊 Smile Master"]
        assert check({"Happy_count": 10})
        assert not check({"Happy_count": 5})

    def test_poker_face_badge(self) -> None:
        check = BADGES["😐 Poker Face"]
        assert check({"Neutral_count": 10})
        assert not check({"Neutral_count": 3})

    def test_grand_master_badge(self) -> None:
        check = BADGES["🏆 Grand Master"]
        assert check({"total_score": 200})
        assert not check({"total_score": 100})

    def test_on_fire_badge(self) -> None:
        check = BADGES["🔥 On Fire"]
        assert check({"max_streak": 5})
        assert not check({"max_streak": 2})


# ── Emotion Utils Pure Functions ────────────────────────────────────────────


class TestComputePositivityScore:
    """Tests for the emotion_utils positivity score (different from config version)."""

    def test_all_happy(self) -> None:
        from utils.emotion_utils import compute_positivity_score

        probs = np.array([0, 0, 0, 1, 0, 0, 0])
        score = compute_positivity_score(probs)
        assert abs(score - 1.0) < 1e-6

    def test_all_angry(self) -> None:
        from utils.emotion_utils import compute_positivity_score

        probs = np.array([1, 0, 0, 0, 0, 0, 0])
        score = compute_positivity_score(probs)
        assert abs(score - (-1.0)) < 1e-6


class TestApplyTemporalSmoothing:
    """Tests for temporal smoothing of predictions."""

    def test_short_history_returns_new(self) -> None:
        from utils.emotion_utils import apply_temporal_smoothing

        history = []
        new_pred = {
            "emotion": "Happy",
            "confidence": 0.9,
            "probabilities": [0, 0, 0, 0.9, 0.1, 0, 0],
        }
        result = apply_temporal_smoothing(history, new_pred, window=5)
        assert result["emotion"] == "Happy"

    def test_smoothing_reduces_flickering(self) -> None:
        from utils.emotion_utils import apply_temporal_smoothing

        history = [
            {"emotion": "Happy", "confidence": 0.8, "probabilities": [0, 0, 0, 0.8, 0.2, 0, 0]},
            {"emotion": "Happy", "confidence": 0.7, "probabilities": [0, 0, 0, 0.7, 0.3, 0, 0]},
            {"emotion": "Happy", "confidence": 0.9, "probabilities": [0, 0, 0, 0.9, 0.1, 0, 0]},
        ]
        new_pred = {
            "emotion": "Sad",
            "confidence": 0.6,
            "probabilities": [0, 0, 0, 0.3, 0.1, 0.6, 0],
        }
        result = apply_temporal_smoothing(history, new_pred, window=5)
        # Should still be Happy because of the 3 previous Happy frames
        assert result["emotion"] == "Happy"

    def test_window_limit(self) -> None:
        from utils.emotion_utils import apply_temporal_smoothing

        history = [
            {"emotion": "Happy", "confidence": 0.8, "probabilities": [0, 0, 0, 0.8, 0.2, 0, 0]}
            for _ in range(10)
        ]
        new_pred = {
            "emotion": "Sad",
            "confidence": 0.9,
            "probabilities": [0, 0, 0, 0, 0, 0.9, 0],
        }
        apply_temporal_smoothing(history, new_pred, window=5)
        # Function pops only one item per call when over window
        assert len(history) <= 10


class TestGenerateEmotionSummary:
    """Tests for emotion summary generation."""

    def test_empty_results(self) -> None:
        from utils.emotion_utils import generate_emotion_summary

        assert generate_emotion_summary([]) == "No faces detected."

    def test_single_result(self) -> None:
        from utils.emotion_utils import generate_emotion_summary

        results = [{"emotion": "Happy", "confidence": 0.9}]
        summary = generate_emotion_summary(results)
        assert "Happy" in summary
        assert "90.0%" in summary

    def test_multiple_results(self) -> None:
        from utils.emotion_utils import generate_emotion_summary

        results = [
            {"emotion": "Happy", "confidence": 0.8},
            {"emotion": "Happy", "confidence": 0.7},
            {"emotion": "Neutral", "confidence": 0.6},
        ]
        summary = generate_emotion_summary(results)
        assert "67%" in summary  # 2/3 Happy
        assert "33%" in summary  # 1/3 Neutral


class TestPreprocessFace:
    """Tests for face preprocessing."""

    def test_output_shape(self) -> None:
        import cv2

        # Mock cv2.resize to return a proper array
        cv2.resize = lambda roi, size, **kw: np.random.randint(0, 255, size, dtype=np.uint8)
        from utils.emotion_utils import preprocess_face

        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.shape == (1, 48, 48, 1)

    def test_output_dtype(self) -> None:
        import cv2

        cv2.resize = lambda roi, size, **kw: np.random.randint(0, 255, size, dtype=np.uint8)
        from utils.emotion_utils import preprocess_face

        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.dtype == np.float32

    def test_output_normalized(self) -> None:
        import cv2

        cv2.resize = lambda roi, size, **kw: np.full(size, 255, dtype=np.uint8)
        from utils.emotion_utils import preprocess_face

        face = np.full((100, 100), 255, dtype=np.uint8)
        result = preprocess_face(face)
        assert result.max() <= 1.0
        assert result.min() >= 0.0
