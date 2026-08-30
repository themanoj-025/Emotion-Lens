"""Tests for model utility functions."""
import sys
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before import (PIL NOT mocked — needed by other tests)
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

from utils.model_utils import (
    EMOTION_CONFIG,
    EMOTIONS,
    MODEL_PATH,
    get_model_summary,
    is_model_available,
    load_face_cascade,
)


class TestEmotions:
    """Tests for emotion constants and configuration."""

    def test_emotions_is_list_of_7(self) -> None:
        assert isinstance(EMOTIONS, list)
        assert len(EMOTIONS) == 7

    def test_emotions_are_strings(self) -> None:
        for e in EMOTIONS:
            assert isinstance(e, str)

    def test_emotion_config_has_all_emotions(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG

    def test_emotion_config_keys_have_required_fields(self) -> None:
        for emotion, config in EMOTION_CONFIG.items():
            assert "color" in config
            assert "emoji" in config
            assert "bg" in config


class TestModelConstants:
    """Tests for module-level constants."""

    def test_model_path_is_string(self) -> None:
        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".h5")

    def test_emotions_colors_are_hex(self) -> None:
        for emotion, config in EMOTION_CONFIG.items():
            assert config["color"].startswith("#")
            assert len(config["color"]) == 7


class TestLoadFaceCascade:
    """Tests for face cascade loading."""

    def test_returns_callable(self) -> None:
        result = load_face_cascade()
        assert result is not None


class TestIsModelAvailable:
    """Tests for is_model_available function."""

    def test_returns_false_when_no_model(self) -> None:
        with patch("os.path.exists", return_value=False):
            assert is_model_available() is False


class TestGetModelSummary:
    """Tests for get_model_summary function."""

    def test_returns_tuple(self) -> None:
        mock_model = MagicMock()
        mock_model.layers = []
        mock_model.count_params.return_value = 1000000
        result = get_model_summary(mock_model)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_layers_and_params(self) -> None:
        mock_layer = MagicMock()
        mock_layer.name = "dense"
        mock_layer.__class__ = type("Dense", (), {})
        mock_layer.output_shape = (None, 7)
        mock_layer.count_params.return_value = 50000
        mock_layer.trainable = True

        mock_model = MagicMock()
        mock_model.layers = [mock_layer]
        mock_model.count_params.return_value = 50000
        layers_info, params = get_model_summary(mock_model)
        assert isinstance(layers_info, list)
        assert isinstance(params, dict)
        assert "total" in params


class TestModelUtilsIntegration:
    """Integration tests for model utilities."""

    def test_emotion_config_complete(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG
            config = EMOTION_CONFIG[emotion]
            assert config["color"].startswith("#")
            assert len(config["emoji"]) > 0

    def test_model_path_ends_with_h5(self) -> None:
        assert MODEL_PATH.endswith(".h5")

    def test_emotion_list_matches_config(self) -> None:
        assert set(EMOTIONS) == set(EMOTION_CONFIG.keys())
