"""Tests for face anonymization utility."""

import numpy as np

from utils.emotion_utils import anonymize_faces


class TestAnonymizeFaces:
    """Tests for anonymize_faces."""

    def test_returns_same_shape(self) -> None:
        # Create a blank image (no faces)
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
