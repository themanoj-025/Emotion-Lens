"""Tests for session management utilities."""

import pytest

from utils.session_utils import init_session_state, get_session_history, add_to_history


class TestInitSessionState:
    """Tests for init_session_state."""

    def test_creates_default_keys(self):
        state = {}
        init_session_state(state)
        assert "predictions" in state
        assert "session_start" in state


class TestGetSessionHistory:
    """Tests for get_session_history."""

    def test_empty_history(self):
        state = {"predictions": []}
        history = get_session_history(state)
        assert history == []

    def test_with_history(self):
        state = {"predictions": [{"emotion": "Happy"}]}
        history = get_session_history(state)
        assert len(history) == 1


class TestAddToHistory:
    """Tests for add_to_history."""

    def test_adds_prediction(self):
        state = {"predictions": []}
        add_to_history(state, {"emotion": "Happy", "confidence": 0.9})
        assert len(state["predictions"]) == 1
