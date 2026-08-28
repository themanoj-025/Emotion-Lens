"""Tests for webcam inference module."""

import pytest
from unittest.mock import MagicMock, patch

from webcam_inference import WebcamProcessor


class TestWebcamProcessor:
    """Tests for WebcamProcessor."""

    def test_init(self):
        processor = WebcamProcessor()
        assert processor is not None
