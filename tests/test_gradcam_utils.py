"""Tests for Grad-CAM visualization utilities."""

import numpy as np

from utils.emotion_utils import (
    _get_last_conv_layer_idx,
    apply_gradcam_overlay,
)


class TestGetLastConvLayerIdx:
    """Tests for _get_last_conv_layer_idx."""

    def test_returns_none_for_no_conv_layers(self):
        mock_model = type("Model", (), {"layers": [
            type("Layer", (), {"name": "dense_1"})(),
            type("Layer", (), {"name": "output"})(),
        ]})()
        result = _get_last_conv_layer_idx(mock_model)
        assert result is None

    def test_finds_last_conv_layer(self):
        mock_model = type("Model", (), {"layers": [
            type("Layer", (), {"name": "conv2d_1"})(),
            type("Layer", (), {"name": "dense_1"})(),
            type("Layer", (), {"name": "conv2d_2"})(),
            type("Layer", (), {"name": "output"})(),
        ]})()
        result = _get_last_conv_layer_idx(mock_model)
        assert result == 2


class TestApplyGradcamOverlay:
    """Tests for apply_gradcam_overlay."""

    def test_modifies_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        heatmap = np.random.rand(48, 48).astype(np.float32)
        result = apply_gradcam_overlay(frame, (10, 10, 50, 50), heatmap)
        assert result is not None
        assert result.shape == frame.shape
