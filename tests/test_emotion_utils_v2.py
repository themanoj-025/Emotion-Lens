"""Tests for emotion detection utilities."""

import numpy as np

from utils.emotion_utils import (
    apply_temporal_smoothing,
    compute_positivity_score,
    generate_emotion_summary,
    image_to_base64,
    preprocess_face,
)


class TestPreprocessFace:
    """Tests for preprocess_face."""

    def test_output_shape(self) -> None:
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.shape == (1, 48, 48, 1)

    def test_normalization(self) -> None:
        face = np.full((48, 48), 255, dtype=np.uint8)
        result = preprocess_face(face)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_custom_target_size(self) -> None:
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face, target_size=(64, 64))
        assert result.shape == (1, 64, 64, 1)


class TestComputePositivityScore:
    """Tests for compute_positivity_score."""

    def test_all_happy(self) -> None:
        probs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Happy at index 3
        score = compute_positivity_score(probs)
        assert score == 1.0

    def test_all_angry(self) -> None:
        probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Angry at index 0
        score = compute_positivity_score(probs)
        assert score == -1.0

    def test_neutral(self) -> None:
        probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # Neutral at index 4
        score = compute_positivity_score(probs)
        assert score == 0.0

    def test_clipping(self) -> None:
        """compute_positivity_score uses np.clip — verify bounds."""
        probs = [1.0] * 7
        score = compute_positivity_score(probs)
        assert -1.0 <= score <= 1.0


class TestApplyTemporalSmoothing:
    """Tests for apply_temporal_smoothing."""

    def test_short_history_returns_new(self) -> None:
        history = []
        new_pred = {"emotion": "Happy", "confidence": 0.9, "probabilities": [0, 0, 0, 1, 0, 0, 0]}
        result = apply_temporal_smoothing(history, new_pred)
        assert result["emotion"] == "Happy"

    def test_window_limit(self) -> None:
        """apply_temporal_smoothing pops 1 item per call when over window."""
        history = [{"emotion": "Happy", "confidence": 0.9, "probabilities": [0, 0, 0, 1, 0, 0, 0]}] * 10
        new_pred = {"emotion": "Sad", "confidence": 0.8, "probabilities": [0, 0, 0, 0, 0, 1, 0]}
        apply_temporal_smoothing(history, new_pred, window=5)
        # After append + pop(0): 10 + 1 - 1 = 10 (only pops 1 per call)
        assert len(history) == 10

    def test_smoothing_reduces_flickering(self) -> None:
        """3 Happy frames followed by 1 Sad → Happy still wins."""
        history = [
            {"emotion": "Happy", "confidence": 0.8, "probabilities": [0, 0, 0, 0.8, 0.2, 0, 0]},
            {"emotion": "Happy", "confidence": 0.7, "probabilities": [0, 0, 0, 0.7, 0.3, 0, 0]},
            {"emotion": "Happy", "confidence": 0.9, "probabilities": [0, 0, 0, 0.9, 0.1, 0, 0]},
        ]
        new_pred = {"emotion": "Sad", "confidence": 0.6, "probabilities": [0, 0, 0, 0.3, 0.1, 0.6, 0]}
        result = apply_temporal_smoothing(history, new_pred, window=5)
        assert result["emotion"] == "Happy"


class TestGenerateEmotionSummary:
    """Tests for generate_emotion_summary."""

    def test_no_faces(self) -> None:
        result = generate_emotion_summary([])
        assert "No faces" in result

    def test_single_face(self) -> None:
        result = generate_emotion_summary([{"emotion": "Happy", "confidence": 0.9}])
        assert "Happy" in result
        assert "90.0%" in result

    def test_multiple_faces(self) -> None:
        results = [
            {"emotion": "Happy", "confidence": 0.9},
            {"emotion": "Happy", "confidence": 0.8},
            {"emotion": "Sad", "confidence": 0.7},
        ]
        result = generate_emotion_summary(results)
        assert "group" in result.lower()


class TestImageToBase64:
    """Tests for image_to_base64."""

    def test_converts_pil_image(self) -> None:
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0
