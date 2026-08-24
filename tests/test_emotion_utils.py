"""Tests for EmotionLens emotion_utils — preprocessing, scoring, smoothing, and utilities."""

import numpy as np
import pytest

# Check if tensorflow/cv2 are available
try:
    import cv2
    from utils.emotion_utils import (
        anonymize_faces,
        apply_temporal_smoothing,
        compute_positivity_score,
        generate_emotion_summary,
        image_to_base64,
        preprocess_face,
    )
    HAS_TF = True
except (ImportError, ModuleNotFoundError):
    HAS_TF = False


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestPreprocessFace:
    """Tests for the face preprocessing pipeline."""

    def test_returns_correct_shape(self):
        fake_face = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = preprocess_face(fake_face)
        assert result.shape == (1, 48, 48, 1)

    def test_output_is_float32(self):
        fake_face = np.ones((48, 48), dtype=np.uint8) * 128
        result = preprocess_face(fake_face)
        assert result.dtype == np.float32

    def test_values_scaled_to_0_1(self):
        fake_face = np.full((48, 48), 255, dtype=np.uint8)
        result = preprocess_face(fake_face)
        assert result.max() == pytest.approx(1.0, abs=1e-6)
        assert result.min() == pytest.approx(1.0, abs=1e-6)

    def test_custom_target_size(self):
        fake_face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(fake_face, target_size=(32, 32))
        assert result.shape == (1, 32, 32, 1)

    def test_zero_image(self):
        fake_face = np.zeros((48, 48), dtype=np.uint8)
        result = preprocess_face(fake_face)
        assert result.sum() == 0.0


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestComputePositivityScore:
    """Tests for valence/positivity scoring."""

    def test_all_happy_is_positive(self):
        probs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        score = compute_positivity_score(probs)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_all_angry_is_negative(self):
        probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        score = compute_positivity_score(probs)
        assert score == pytest.approx(-1.0, abs=1e-6)

    def test_neutral_is_zero(self):
        probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        score = compute_positivity_score(probs)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_bounded_between_minus1_and_1(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            probs = rng.dirichlet(np.ones(7)).tolist()
            score = compute_positivity_score(probs)
            assert -1.0 <= score <= 1.0

    def test_mixed_distribution(self):
        probs = [1.0 / 7] * 7
        score = compute_positivity_score(probs)
        assert -0.2 <= score <= 0.2


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestApplyTemporalSmoothing:
    """Tests for rolling-average temporal smoothing."""

    def test_returns_same_prediction_with_few_frames(self):
        history = []
        pred = {"emotion": "Happy", "confidence": 0.9, "probabilities": [0.0] * 7}
        pred["probabilities"][3] = 0.9
        result = apply_temporal_smoothing(history, pred, window=5)
        assert result["emotion"] == "Happy"

    def test_smoothing_reduces_flickering(self):
        history = []
        happy_pred = {"emotion": "Happy", "confidence": 0.8, "probabilities": [0.0] * 7}
        happy_pred["probabilities"][3] = 0.8
        for _ in range(5):
            apply_temporal_smoothing(history, happy_pred, window=5)
        sad_pred = {"emotion": "Sad", "confidence": 0.9, "probabilities": [0.0] * 7}
        sad_pred["probabilities"][5] = 0.9
        result = apply_temporal_smoothing(history, sad_pred, window=5)
        assert result["emotion"] == "Happy"

    def test_window_limits_history(self):
        history = []
        pred = {"emotion": "Happy", "confidence": 0.5, "probabilities": [1.0 / 7] * 7}
        for _ in range(20):
            apply_temporal_smoothing(history, pred, window=5)
        assert len(history) <= 5


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestGenerateEmotionSummary:
    """Tests for group/multi-face summary generation."""

    def test_empty_results(self):
        assert generate_emotion_summary([]) == "No faces detected."

    def test_single_face(self):
        results = [{"emotion": "Happy", "confidence": 0.95, "probabilities": [0.0] * 7}]
        summary = generate_emotion_summary(results)
        assert "Happy" in summary
        assert "95.0%" in summary

    def test_multiple_faces_groups_by_emotion(self):
        results = [
            {"emotion": "Happy", "confidence": 0.8, "probabilities": [0.0] * 7},
            {"emotion": "Happy", "confidence": 0.7, "probabilities": [0.0] * 7},
            {"emotion": "Sad", "confidence": 0.6, "probabilities": [0.0] * 7},
        ]
        summary = generate_emotion_summary(results)
        assert "Happy" in summary
        assert "Sad" in summary
        assert "67%" in summary


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestImageToBase64:
    """Tests for base64 encoding utility."""

    def test_returns_nonempty_string(self):
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        result = image_to_base64(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_base64(self):
        import base64
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="blue")
        result = image_to_base64(img)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


@pytest.mark.skipif(not HAS_TF, reason="tensorflow/cv2 not installed")
class TestAnonymizeFaces:
    """Tests for face anonymization (blur/pixelate)."""

    def test_no_faces_returns_unchanged(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = anonymize_faces(img)
        assert result.shape == img.shape

    def test_returns_same_array_in_place(self):
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        result = anonymize_faces(img)
        assert result is img

    def test_pixelate_mode(self):
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        result = anonymize_faces(img, pixelate=True)
        assert result.shape == img.shape


class TestEmotionConfigConsistency:
    """Cross-module consistency checks (config-only, no TF needed)."""

    def test_emotion_config_has_all_7_emotions(self):
        from utils.config import EMOTION_CONFIG, EMOTIONS

        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG, f"Missing config for {emotion}"

    def test_emotion_config_color_format(self):
        from utils.config import EMOTION_CONFIG

        for emotion, cfg in EMOTION_CONFIG.items():
            assert cfg["color"].startswith("#"), f"{emotion} color missing #"
            assert len(cfg["color"]) == 7, f"{emotion} color should be 7 chars"

    def test_mood_music_has_all_emotions(self):
        from utils.config import EMOTIONS, MOOD_MUSIC

        for emotion in EMOTIONS:
            assert emotion in MOOD_MUSIC, f"Missing mood music for {emotion}"
            assert "genre" in MOOD_MUSIC[emotion]
            assert "query" in MOOD_MUSIC[emotion]
