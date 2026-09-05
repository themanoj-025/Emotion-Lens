"""Tests for face anonymization utility."""

import sys
from unittest.mock import MagicMock

import numpy as np

# Mock tensorflow and streamlit before importing emotion_utils
# (emotion_utils -> model_utils -> tensorflow at module level)
sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())
sys.modules.setdefault("streamlit", MagicMock())

from utils.emotion_utils import anonymize_faces


class TestAnonymizeFaces:
    """Tests for anonymize_faces."""

    def test_returns_same_shape(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = anonymize_faces(image)
        assert result.shape == image.shape

    def test_blur_mode(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = anonymize_faces(image, pixelate=False)
        assert result is not None

    def test_pixelate_mode(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = anonymize_faces(image, pixelate=True)
        assert result is not None

    def test_custom_kernel_size(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = anonymize_faces(image, kernel_size=(51, 51))
        assert result is not None

    def test_no_faces_returns_unchanged(self) -> None:
        """With no faces detected, image should be returned unchanged."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        original = image.copy()
        result = anonymize_faces(image)
        np.testing.assert_array_equal(result, original)

    def test_odd_kernel_size_enforced(self) -> None:
        """Even kernel sizes should be made odd (OpenCV requirement)."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Even kernel size should still work (made odd internally)
        result = anonymize_faces(image, kernel_size=(50, 50))
        assert result.shape == image.shape
