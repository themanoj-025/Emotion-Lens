"""Temporal smoothing to reduce flickering in live detection."""
from __future__ import annotations
from collections import deque
import numpy as np
from utils.config import SMOOTHING_WINDOW, EMOTIONS


class EmotionSmoother:
    """Rolling-average smoother for live emotion probability arrays.

    Maintains a sliding window of recent probability arrays and returns
    the element-wise mean, which stabilizes predictions and reduces flickering.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.window = window
        self._buffer: deque[list[float]] = deque(maxlen=window)

    def update(self, probs: list[float]) -> list[float]:
        """Add new probabilities; return smoothed array.

        Args:
            probs: List of 7 probability values.

        Returns:
            Smoothed list of 7 probability values.
        """
        self._buffer.append(probs)
        arr = np.array(self._buffer)
        smoothed = arr.mean(axis=0).tolist()
        return smoothed

    def smoothed_emotion(self, probs: list[float]) -> tuple[str, float, list[float]]:
        """Update buffer; return (emotion, confidence, all_smoothed_probs).

        Args:
            probs: Raw probability array of length 7.

        Returns:
            Tuple of (emotion_label, confidence_of_top, all_smoothed_probs).
        """
        smoothed = self.update(probs)
        idx = int(np.argmax(smoothed))
        return EMOTIONS[idx], smoothed[idx], smoothed

    def reset(self):
        """Clear the smoothing buffer."""
        self._buffer.clear()
