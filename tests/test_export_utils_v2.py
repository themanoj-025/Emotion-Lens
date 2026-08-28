"""Tests for export utilities."""

import json

import pandas as pd
import pytest

from utils.export_utils import (
    export_predictions_csv,
    export_predictions_json,
    export_session_report,
    predictions_to_dataframe,
)


@pytest.fixture
def sample_predictions():
    return [
        {
            "emotion": "Happy",
            "confidence": 0.9,
            "probabilities": [0.0, 0.0, 0.0, 0.9, 0.05, 0.05, 0.0],
            "timestamp": "2025-01-01 12:00:00",
            "source": "live",
        },
        {
            "emotion": "Sad",
            "confidence": 0.7,
            "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
            "timestamp": "2025-01-01 12:01:00",
            "source": "live",
        },
    ]


class TestExportPredictionsCsv:
    """Tests for export_predictions_csv."""

    def test_empty_returns_none(self):
        assert export_predictions_csv([]) is None

    def test_csv_content(self, sample_predictions):
        csv = export_predictions_csv(sample_predictions)
        assert csv is not None
        assert "emotion" in csv
        assert "Happy" in csv
        assert "Sad" in csv

    def test_csv_has_all_emotions(self, sample_predictions):
        from utils.config import EMOTIONS
        csv = export_predictions_csv(sample_predictions)
        for e in EMOTIONS:
            assert e in csv


class TestExportPredictionsJson:
    """Tests for export_predictions_json."""

    def test_empty_returns_none(self):
        assert export_predictions_json([]) is None

    def test_json_content(self, sample_predictions):
        result = export_predictions_json(sample_predictions)
        parsed = json.loads(result)
        assert len(parsed) == 2


class TestExportSessionReport:
    """Tests for export_session_report."""

    def test_empty_returns_none(self):
        assert export_session_report([]) is None

    def test_report_content(self, sample_predictions):
        report = export_session_report(sample_predictions)
        assert "Total Predictions: 2" in report
        assert "Dominant Emotion" in report


class TestPredictionsToDataframe:
    """Tests for predictions_to_dataframe."""

    def test_empty_returns_empty_df(self):
        df = predictions_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_creates_dataframe(self, sample_predictions):
        df = predictions_to_dataframe(sample_predictions)
        assert len(df) == 2
        assert "positivity_score" in df.columns
