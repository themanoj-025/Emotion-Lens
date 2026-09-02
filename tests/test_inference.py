import pytest

pytestmark = pytest.mark.unit

"""Tests for inference module."""

from unittest.mock import MagicMock, patch

from inference import load_face_cascade, load_model

pytestmark = pytest.mark.slow
class TestLoadModel:
    """Tests for load_model."""

    def test_load_model(self) -> None:
        with patch("inference.tf.keras.models.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            model = load_model("dummy_path.h5")
            assert model is not None

    def test_load_model_returns_none_on_error(self) -> None:
        with patch("inference.tf.keras.models.load_model", side_effect=OSError("not found")):
            model = load_model("nonexistent.h5")
            assert model is None


class TestLoadFaceCascade:
    """Tests for load_face_cascade."""

    def test_loads_default_cascade(self) -> None:
        cascade = load_face_cascade()
        assert cascade is not None
