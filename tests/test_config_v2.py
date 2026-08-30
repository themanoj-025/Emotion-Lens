"""Tests for emotion configuration constants."""


from utils.config import EMOTIONS, SMOOTHING_WINDOW, positivity_score


class TestEmotions:
    """Tests for EMOTIONS constant."""

    def test_has_7_emotions(self) -> None:
        assert len(EMOTIONS) == 7

    def test_emotion_names(self) -> None:
        expected = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}
        assert set(EMOTIONS) == expected


class TestPositivityScore:
    """Tests for positivity_score function."""

    def test_happy_positive(self) -> None:
        probs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        assert positivity_score(probs) > 0

    def test_sad_negative(self) -> None:
        probs = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        assert positivity_score(probs) < 0

    def test_neutral_zero(self) -> None:
        probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        assert positivity_score(probs) == 0.0

    def test_clipping(self) -> None:
        probs = [1.0] * 7
        score = positivity_score(probs)
        assert -1.0 <= score <= 1.0


class TestSmoothingWindow:
    """Tests for SMOOTHING_WINDOW constant."""

    def test_default_window(self) -> None:
        assert SMOOTHING_WINDOW == 5
        assert isinstance(SMOOTHING_WINDOW, int)
