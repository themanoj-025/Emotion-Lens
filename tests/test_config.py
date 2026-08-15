"""Tests for EmotionLens 🎭 config module — pure-logic constants and helpers."""

from utils.config import (
    EMOTIONS,
    EMOTION_CONFIG,
    BADGES,
    MOOD_MUSIC,
    emotion_index,
    positivity_score,
)


def test_emotions_follow_fer2013_order():
    """EMOTIONS must match FER2013 training order (7 classes)."""
    assert EMOTIONS == [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    ]


def test_emotion_config_covers_all_emotions():
    """Every emotion has a color, bg, emoji, valence and arousal."""
    for emotion in EMOTIONS:
        cfg = EMOTION_CONFIG[emotion]
        assert cfg["color"].startswith("#")
        assert cfg["emoji"]
        assert -1.0 <= cfg["valence"] <= 1.0
        # arousal: -1.0 (calm/sad) to 1.0 (excited) — Sad is -0.4
        assert -1.0 <= cfg["arousal"] <= 1.0


def test_mood_music_covers_all_emotions():
    """Every emotion maps to a music genre + search query."""
    for emotion in EMOTIONS:
        assert "genre" in MOOD_MUSIC[emotion]
        assert "query" in MOOD_MUSIC[emotion]


def test_positivity_score_happy_positive():
    probs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # 100% Happy
    assert positivity_score(probs) == 1.0


def test_positivity_score_angry_negative():
    probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 100% Angry
    assert positivity_score(probs) == -1.0


def test_positivity_score_neutral_zero():
    probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # 100% Neutral
    assert positivity_score(probs) == 0.0


def test_positivity_score_bounds():
    """Mixed distributions must stay within [-1, 1]."""
    import numpy as np

    for _ in range(50):
        probs = np.random.dirichlet(np.ones(7)).tolist()
        score = positivity_score(probs)
        assert -1.0 <= score <= 1.0


def test_emotion_index():
    assert emotion_index("Happy") == 3
    assert emotion_index("Angry") == 0
    assert emotion_index("Surprise") == 6


def test_badges_all_callable():
    """Every badge is a lambda evaluating session state dicts."""
    for name, predicate in BADGES.items():
        assert callable(predicate)
        assert name  # non-empty badge name


def test_badge_conditions():
    assert BADGES["😊 Smile Master"]({"Happy_count": 10})
    assert not BADGES["😊 Smile Master"]({"Happy_count": 3})
    assert BADGES["🏆 Grand Master"]({"total_score": 250})
    assert not BADGES["🏆 Grand Master"]({"total_score": 50})
