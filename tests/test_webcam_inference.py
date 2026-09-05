"""Tests for webcam inference module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Mock tensorflow before importing webcam_inference
sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())
sys.modules.setdefault("tensorflow.keras.preprocessing", MagicMock())
sys.modules.setdefault("tensorflow.keras.preprocessing.image", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from webcam_inference import EMOTIONS, load_emotion_model


class TestEmotions:
    """Tests for EMOTIONS constant."""

    def test_has_7_emotions(self) -> None:
        assert len(EMOTIONS) == 7

    def test_matches_fer2013_order(self) -> None:
        assert EMOTIONS == [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]


class TestLoadEmotionModel:
    """Tests for load_emotion_model."""

    @patch("webcam_inference.load_model")
    def test_loads_model_successfully(self, mock_load) -> None:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        result = load_emotion_model("test_model.h5")
        assert result is not None
        mock_load.assert_called_once_with("test_model.h5")

    @patch("webcam_inference.load_model", side_effect=OSError("not found"))
    def test_returns_none_on_error(self, mock_load) -> None:
        result = load_emotion_model("nonexistent.h5")
        assert result is None

    @patch("webcam_inference.load_model")
    def test_default_path(self, mock_load) -> None:
        mock_load.return_value = MagicMock()
        load_emotion_model()
        mock_load.assert_called_once_with("emotion_model.h5")
