"""Tests for session management utilities."""

import sys
from unittest.mock import MagicMock

import pytest

# Ensure streamlit is mocked (may already be mocked by other test files)
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

import streamlit as st


class _SessionState(dict):
    """Dict subclass that also supports attribute access (like Streamlit's session_state)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'SessionState' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'SessionState' object has no attribute '{key}'")


@pytest.fixture(autouse=True)
def mock_session_state():
    """Reset streamlit session state mock before each test."""
    st.session_state = _SessionState()
    yield
    st.session_state = _SessionState()


from utils.session_utils import (
    add_prediction,
    export_predictions_csv,
    export_predictions_json,
    format_session_duration,
    get_emotion_distribution,
    get_prediction_dataframe,
    init_session_state,
    reset_session,
)


class TestInitSessionState:
    """Tests for init_session_state."""

    def test_creates_default_keys(self) -> None:
        init_session_state()
        assert "predictions" in st.session_state
        assert "session_start" in st.session_state
        assert "game_score" in st.session_state
        assert "dark_mode" in st.session_state

    def test_idempotent(self) -> None:
        """Calling init twice should not overwrite existing values."""
        init_session_state()
        st.session_state["predictions"] = [{"emotion": "Happy"}]
        init_session_state()
        assert len(st.session_state["predictions"]) == 1

    def test_default_values(self) -> None:
        init_session_state()
        assert st.session_state["predictions"] == []
        assert st.session_state["total_predictions"] == 0
        assert st.session_state["game_score"] == 0
        assert st.session_state["dark_mode"] is True
        assert st.session_state["camera_active"] is False


class TestAddPrediction:
    """Tests for add_prediction."""

    def test_adds_single_prediction(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05])
        assert len(st.session_state["predictions"]) == 1
        assert st.session_state["predictions"][0]["emotion"] == "Happy"
        assert st.session_state["total_predictions"] == 1

    def test_adds_multiple_predictions(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0])
        add_prediction("Sad", 0.7, [0.0, 0.0, 0.0, 0.1, 0.1, 0.7, 0.1])
        assert len(st.session_state["predictions"]) == 2
        assert st.session_state["total_predictions"] == 2

    def test_converts_numpy_array(self) -> None:
        """Should handle numpy arrays for probabilities."""
        import numpy as np

        init_session_state()
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
        add_prediction("Surprise", 0.4, probs)
        assert st.session_state["predictions"][0]["probabilities"] == pytest.approx(
            probs.tolist()
        )

    def test_has_timestamp(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0] * 7)
        assert "timestamp" in st.session_state["predictions"][0]


class TestGetPredictionDataframe:
    """Tests for get_prediction_dataframe."""

    def test_empty_returns_empty_df(self) -> None:
        init_session_state()
        df = get_prediction_dataframe()
        assert df.empty

    def test_with_predictions(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0])
        df = get_prediction_dataframe()
        assert len(df) == 1
        assert "emotion" in df.columns
        assert "timestamp" in df.columns


class TestGetEmotionDistribution:
    """Tests for get_emotion_distribution."""

    def test_empty_returns_empty_dict(self) -> None:
        init_session_state()
        dist = get_emotion_distribution()
        assert dist == {}

    def test_counts_emotions(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0] * 7)
        add_prediction("Happy", 0.8, [0.0] * 7)
        add_prediction("Sad", 0.7, [0.0] * 7)
        dist = get_emotion_distribution()
        assert dist["Happy"] == 2
        assert dist["Sad"] == 1


class TestExportPredictionsCsv:
    """Tests for export_predictions_csv."""

    def test_empty_returns_none(self) -> None:
        init_session_state()
        assert export_predictions_csv() is None

    def test_with_predictions_returns_csv(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0] * 7)
        csv_content = export_predictions_csv()
        assert csv_content is not None
        assert "Happy" in csv_content


class TestExportPredictionsJson:
    """Tests for export_predictions_json."""

    def test_empty_returns_none(self) -> None:
        init_session_state()
        assert export_predictions_json() is None

    def test_with_predictions_returns_json(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0] * 7)
        import json

        json_content = export_predictions_json()
        data = json.loads(json_content)
        assert len(data) == 1
        assert data[0]["emotion"] == "Happy"


class TestFormatSessionDuration:
    """Tests for format_session_duration."""

    def test_returns_time_format(self) -> None:
        init_session_state()
        result = format_session_duration()
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 3

    def test_without_session_start(self) -> None:
        st.session_state = _SessionState()
        result = format_session_duration()
        assert result == "00:00:00"


class TestResetSession:
    """Tests for reset_session."""

    def test_clears_predictions(self) -> None:
        init_session_state()
        add_prediction("Happy", 0.9, [0.0] * 7)
        reset_session()
        assert st.session_state["predictions"] == []

    def test_preserves_preferences(self) -> None:
        init_session_state()
        st.session_state["dark_mode"] = False
        st.session_state["model_path"] = "custom_model.h5"
        reset_session()
        assert st.session_state["dark_mode"] is False
        assert st.session_state["model_path"] == "custom_model.h5"
