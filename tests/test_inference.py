"""Tests for inference module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock tensorflow before importing inference
sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())

from inference import (
    decode_base64_image,
    generate_summary,
    get_model,
    preprocess_face,
    predict_face,
    process_image,
)


class TestGetModel:
    """Tests for get_model."""

    @patch("inference.os.path.exists", return_value=False)
    def test_returns_none_when_model_missing(self, mock_exists) -> None:
        model, cascade = get_model()
        assert model is None
        assert cascade is None

    @patch("inference.os.path.exists", return_value=True)
    @patch("inference.cv2.CascadeClassifier")
    def test_loads_cascade(self, mock_cascade_cls, mock_exists) -> None:
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade_cls.return_value = mock_cascade
        # Reset global state
        import inference

        inference._model = MagicMock()
        inference._face_cascade = None
        model, cascade = get_model()
        assert cascade is not None


class TestPreprocessFace:
    """Tests for preprocess_face."""

    def test_output_shape(self) -> None:
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.shape == (1, 48, 48, 1)

    def test_output_dtype(self) -> None:
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.dtype == np.float32

    def test_output_normalized(self) -> None:
        face = np.full((100, 100), 255, dtype=np.uint8)
        result = preprocess_face(face)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_small_face_upscaled(self) -> None:
        face = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        result = preprocess_face(face)
        assert result.shape == (1, 48, 48, 1)


class TestPredictFace:
    """Tests for predict_face."""

    def test_returns_emotion_and_confidence(self) -> None:
        mock_model = MagicMock()
        # Return probabilities: 7 emotions, Happy (index 3) highest
        probs = np.array([0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.05])
        mock_model.predict.return_value = np.array([probs])

        face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        emotion, confidence, prob_dict = predict_face(mock_model, face)

        assert emotion == "Happy"
        assert confidence == pytest.approx(0.7, abs=0.01)
        assert len(prob_dict) == 7

    def test_returns_all_seven_emotions(self) -> None:
        mock_model = MagicMock()
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
        mock_model.predict.return_value = np.array([probs])

        face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        _, _, prob_dict = predict_face(mock_model, face)

        expected_keys = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}
        assert set(prob_dict.keys()) == expected_keys


class TestGenerateSummary:
    """Tests for generate_summary."""

    def test_empty_results(self) -> None:
        assert generate_summary([]) == "No faces detected."

    def test_single_result(self) -> None:
        from api_models import EmotionResult

        result = EmotionResult(emotion="Happy", confidence=0.9, probabilities={})
        summary = generate_summary([result])
        assert "Happy" in summary
        assert "90.0%" in summary

    def test_multiple_results_same_emotion(self) -> None:
        from api_models import EmotionResult

        results = [
            EmotionResult(emotion="Happy", confidence=0.8, probabilities={}),
            EmotionResult(emotion="Happy", confidence=0.7, probabilities={}),
        ]
        summary = generate_summary(results)
        assert "Happy" in summary
        assert "100%" in summary

    def test_multiple_results_mixed(self) -> None:
        from api_models import EmotionResult

        results = [
            EmotionResult(emotion="Happy", confidence=0.8, probabilities={}),
            EmotionResult(emotion="Neutral", confidence=0.6, probabilities={}),
        ]
        summary = generate_summary(results)
        assert "Group:" in summary
        assert "50%" in summary


class TestDecodeBase64Image:
    """Tests for decode_base64_image."""

    def test_invalid_base64_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid base64"):
            decode_base64_image("not-valid-base64!!!")

    def test_valid_base64_image(self) -> None:
        import base64

        # Create a minimal valid PNG image
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = decode_base64_image(b64)
        assert result.shape[2] == 3  # BGR has 3 channels

    def test_data_uri_prefix_stripped(self) -> None:
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        result = decode_base64_image(data_uri)
        assert result.shape[2] == 3


class TestProcessImage:
    """Tests for process_image."""

    def test_no_faces_detects_full_image(self) -> None:
        mock_model = MagicMock()
        probs = np.array([0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.05])
        mock_model.predict.return_value = np.array([probs])

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([])

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        results, count = process_image(mock_model, mock_cascade, img)
        assert count == 1
        assert len(results) == 1
