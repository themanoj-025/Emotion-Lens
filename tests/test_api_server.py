"""
Unit tests for Emotion-Lens — FastAPI REST API (api_server.py).

Covers: Pydantic models, summary generation, auth logic.
NOTE: TensorFlow not available in CI, so we mock the model loading.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock TensorFlow before importing api_server
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()

import api_server
from api_server import EmotionResult, HealthResponse, PredictRequest, PredictResponse, generate_summary


# ── Pydantic Model Tests ──────────────────────────────────────────────────

class TestPydanticModels:
    def test_emotion_result(self):
        result = EmotionResult(
            emotion="happy", confidence=0.95, probabilities={"happy": 0.95, "sad": 0.05}
        )
        assert result.emotion == "happy"
        assert result.confidence == 0.95
        assert result.bbox is None

    def test_emotion_result_with_bbox(self):
        result = EmotionResult(
            emotion="sad", confidence=0.8, probabilities={}, bbox=[10, 20, 100, 100]
        )
        assert result.bbox == [10, 20, 100, 100]

    def test_predict_request(self):
        req = PredictRequest(image="base64data")
        assert req.image == "base64data"
        assert req.detect_faces is True

    def test_predict_request_no_detect(self):
        req = PredictRequest(image="data", detect_faces=False)
        assert req.detect_faces is False

    def test_health_response(self):
        resp = HealthResponse(
            status="healthy", model_loaded=True, model_path="/path", emotions=["happy"]
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_predict_response(self):
        resp = PredictResponse(
            success=True, faces_detected=1, results=[], summary="test", processing_time_ms=10.5
        )
        assert resp.success is True
        assert resp.processing_time_ms == 10.5


# ── Generate Summary Tests ────────────────────────────────────────────────

class TestGenerateSummary:
    def test_empty_results(self):
        assert generate_summary([]) == "No faces detected."

    def test_single_face(self):
        results = [EmotionResult(emotion="happy", confidence=0.9, probabilities={})]
        summary = generate_summary(results)
        assert "happy" in summary
        assert "90.0%" in summary

    def test_multiple_faces_same_emotion(self):
        results = [
            EmotionResult(emotion="happy", confidence=0.9, probabilities={}),
            EmotionResult(emotion="happy", confidence=0.8, probabilities={}),
        ]
        summary = generate_summary(results)
        assert "happy" in summary
        assert "100%" in summary

    def test_multiple_faces_mixed(self):
        results = [
            EmotionResult(emotion="happy", confidence=0.9, probabilities={}),
            EmotionResult(emotion="sad", confidence=0.7, probabilities={}),
        ]
        summary = generate_summary(results)
        assert "Group:" in summary
        assert "happy" in summary
        assert "sad" in summary


# ── API Key Auth Tests ────────────────────────────────────────────────────

class TestAPIKeyAuth:
    def test_no_key_allows_open_access(self):
        with patch.object(api_server, "API_KEY", ""):
            result = api_server.verify_api_key(credentials=None)
            assert result is True

    def test_rejects_wrong_key(self):
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        with patch.object(api_server, "API_KEY", "correct-key"):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                api_server.verify_api_key(credentials=creds)
            assert exc_info.value.status_code == 403

    def test_rejects_missing_credentials(self):
        from fastapi import HTTPException
        with patch.object(api_server, "API_KEY", "some-key"):
            with pytest.raises(HTTPException) as exc_info:
                api_server.verify_api_key(credentials=None)
            assert exc_info.value.status_code == 401

    def test_accepts_correct_key(self):
        from fastapi.security import HTTPAuthorizationCredentials
        with patch.object(api_server, "API_KEY", "my-secret"):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="my-secret")
            result = api_server.verify_api_key(credentials=creds)
            assert result is True
