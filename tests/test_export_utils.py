"""Tests for EmotionLens 🎭 export utilities — CSV/JSON/report exports."""

import csv
import io
import json

from utils.export_utils import (
    export_predictions_csv,
    export_predictions_json,
    export_session_report,
    predictions_to_dataframe,
)

SAMPLE_PREDICTIONS = [
    {
        "emotion": "Happy",
        "confidence": 0.85,
        "probabilities": [0.01, 0.01, 0.01, 0.85, 0.05, 0.04, 0.03],
        "timestamp": "2026-01-01T10:00:00",
        "source": "image",
    },
    {
        "emotion": "Neutral",
        "confidence": 0.72,
        "probabilities": [0.02, 0.01, 0.05, 0.10, 0.72, 0.06, 0.04],
        "timestamp": "2026-01-01T10:00:01",
        "source": "image",
    },
]


def test_csv_empty_returns_none():
    assert export_predictions_csv([]) is None


def test_csv_header_and_rows():
    content = export_predictions_csv(SAMPLE_PREDICTIONS)
    assert content is not None
    rows = list(csv.DictReader(io.StringIO(content)))
    assert len(rows) == 2
    assert rows[0]["emotion"] == "Happy"
    assert rows[0]["source"] == "image"
    # Per-emotion probability columns present
    assert rows[0]["Happy"] == "0.85"
    assert rows[0]["Angry"] == "0.01"
    # positivity_score column computed
    score = float(rows[0]["positivity_score"])
    assert -1.0 <= score <= 1.0


def test_json_empty_returns_none():
    assert export_predictions_json([]) is None


def test_json_roundtrip():
    content = export_predictions_json(SAMPLE_PREDICTIONS)
    assert content is not None
    data = json.loads(content)
    assert len(data) == 2
    assert data[0]["emotion"] == "Happy"


def test_session_report_empty_returns_none():
    assert export_session_report([]) is None


def test_session_report_counts_and_dominant():
    report = export_session_report(SAMPLE_PREDICTIONS)
    assert report is not None
    assert "Total Predictions: 2" in report
    assert "Happy" in report  # dominant emotion
    assert "Neutral" in report


def test_session_report_single_prediction():
    report = export_session_report([SAMPLE_PREDICTIONS[0]])
    assert report is not None
    assert "Total Predictions: 1" in report


def test_dataframe_empty():
    df = predictions_to_dataframe([])
    assert df.empty


def test_dataframe_has_positivity_column():
    df = predictions_to_dataframe(SAMPLE_PREDICTIONS)
    assert len(df) == 2
    assert "positivity_score" in df.columns
    assert df["positivity_score"].between(-1, 1).all()


def test_dataframe_malformed_probs_get_zero():
    preds = [
        {
            "emotion": "Happy",
            "confidence": 0.9,
            "probabilities": [1.0, 0.0],  # wrong length
            "timestamp": "t",
        }
    ]
    df = predictions_to_dataframe(preds)
    assert df.iloc[0]["positivity_score"] == 0
