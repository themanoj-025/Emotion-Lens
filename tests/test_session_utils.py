"""Tests for session management utilities."""


from utils.session_utils import add_to_history, get_session_history, init_session_state


class TestInitSessionState:
    """Tests for init_session_state."""

    def test_creates_default_keys(self) -> None:
        state = {}
        init_session_state(state)
        assert "predictions" in state
        assert "session_start" in state


class TestGetSessionHistory:
    """Tests for get_session_history."""

    def test_empty_history(self) -> None:
        state = {"predictions": []}
        history = get_session_history(state)
        assert history == []

    def test_with_history(self) -> None:
        state = {"predictions": [{"emotion": "Happy"}]}
        history = get_session_history(state)
        assert len(history) == 1


class TestAddToHistory:
    """Tests for add_to_history."""

    def test_adds_prediction(self) -> None:
        state = {"predictions": []}
        add_to_history(state, {"emotion": "Happy", "confidence": 0.9})
        assert len(state["predictions"]) == 1
