"""Integration tests for Emotion-Lens — full HTTP lifecycle through FastAPI.

Tests the complete request-response cycle including middleware, error handling,
multi-endpoint workflows, and OpenAPI schema generation. Uses mocked model
(TensorFlow unavailable in CI) but exercises real HTTP routing.
"""

from __future__ import annotations

import base64
import io
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow
# Mock TensorFlow before importing api_server
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()

import api_server
from api_server import app

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> None:
    """Create a TestClient with model mocked."""
    with patch.object(api_server, "_model", MagicMock()), patch.object(
        api_server, "_face_cascade", MagicMock()
    ):
        c = TestClient(app, raise_server_exceptions=False)
        yield c


@pytest.fixture()
def dummy_b64_image() -> str:
    """Create a valid base64-encoded 48x48 grayscale face image."""
    img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pil_img = io.BytesIO()
    from PIL import Image

    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(pil_img, format="PNG")
    return base64.b64encode(pil_img.getvalue()).decode()


@pytest.fixture()
def dummy_b64_with_prefix(dummy_b64_image) -> str:
    """Base64 image with data URI prefix."""
    return f"data:image/png;base64,{dummy_b64_image}"


# ── Full HTTP Lifecycle ───────────────────────────────────────────────────


class TestHTTPLifecycle:
    """Tests that exercise the full request → middleware → handler → response cycle."""

    def test_root_returns_service_info(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "EmotionLens 🎭 API"
        assert data["version"] == "1.0.0"
        assert "emotions" in data
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) == 7

    def test_health_endpoint_returns_model_status(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "unhealthy")
        assert "model_loaded" in data
        assert "emotions" in data

    def test_health_endpoint_returns_listed_emotions(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        expected = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        assert data["emotions"] == expected


# ── Middleware Behavior ────────────────────────────────────────────────────


class TestMiddleware:
    """Verify security headers, CORS, and rate limiting are applied."""

    def test_security_headers_present(self, client) -> None:
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert response.headers.get("X-XSS-Protection") == "0"

    def test_content_security_policy(self, client) -> None:
        response = client.get("/health")
        csp = response.headers.get("Content-Security-Policy", "")
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_permissions_policy(self, client) -> None:
        response = client.get("/health")
        pp = response.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp


# ── Prediction Endpoints ──────────────────────────────────────────────────


class TestPredictionEndpoints:
    """Integration tests for /api/v1/predict and /api/v1/predict-file."""

    @patch("api_server.predict_face", return_value=("Happy", 0.92, {"Happy": 0.92, "Sad": 0.08}))
    @patch("api_server.get_model", return_value=(MagicMock(), MagicMock()))
    def test_predict_base64_returns_success(self, mock_model, mock_predict, client, dummy_b64_image) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": dummy_b64_image, "detect_faces": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["faces_detected"] >= 1
        assert len(data["results"]) >= 1
        assert data["results"][0]["emotion"] == "Happy"
        assert data["results"][0]["confidence"] == pytest.approx(0.92)
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    @patch("api_server.predict_face", return_value=("Happy", 0.92, {"Happy": 0.92}))
    @patch("api_server.get_model", return_value=(MagicMock(), MagicMock()))
    def test_predict_with_data_uri_prefix(self, mock_model, mock_predict, client, dummy_b64_with_prefix) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": dummy_b64_with_prefix, "detect_faces": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_predict_empty_image_returns_400(self, client) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": "", "detect_faces": True},
        )
        assert response.status_code == 400

    def test_predict_invalid_base64_returns_400(self, client) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": "not-valid-base64!!!", "detect_faces": True},
        )
        assert response.status_code == 400

    def test_predict_missing_image_field_returns_422(self, client) -> None:
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_file_endpoint(self, client) -> None:
        """Test file upload endpoint with a dummy image."""
        img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        response = client.post(
            "/api/v1/predict-file",
            files={"file": ("face.png", buf.tobytes(), "image/png")},
            data={"detect_faces": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_predict_file_rejects_non_image(self, client) -> None:
        response = client.post(
            "/api/v1/predict-file",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


# ── Error Handling Workflows ──────────────────────────────────────────────


class TestErrorHandling:
    """Verify graceful error handling across the API."""

    def test_nonexistent_route_returns_404(self, client) -> None:
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method_returns_405(self, client) -> None:
        response = client.post("/health")
        assert response.status_code == 405

    def test_predict_with_wrong_content_type(self, client) -> None:
        response = client.post(
            "/api/v1/predict",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422

    def test_malformed_json_body(self, client) -> None:
        response = client.post(
            "/api/v1/predict",
            content="{invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ── Multi-Endpoint Workflow ────────────────────────────────────────────────


class TestMultiEndpointWorkflow:
    """Simulate a realistic user session: root → health → predict → health."""

    @patch("api_server.predict_face", return_value=("Neutral", 0.85, {"Neutral": 0.85}))
    @patch("api_server.get_model", return_value=(MagicMock(), MagicMock()))
    def test_full_user_workflow(self, mock_model, mock_predict, client, dummy_b64_image) -> None:
        # Step 1: Discover API
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["version"] == "1.0.0"

        # Step 2: Check health
        health = client.get("/health")
        assert health.status_code == 200

        # Step 3: Make prediction
        predict = client.post(
            "/api/v1/predict",
            json={"image": dummy_b64_image, "detect_faces": True},
        )
        assert predict.status_code == 200
        assert predict.json()["success"] is True

        # Step 4: Check health again (model still loaded)
        health2 = client.get("/health")
        assert health2.status_code == 200

    def test_openapi_schema_is_valid(self, client) -> None:
        """Verify the OpenAPI schema is generated and well-formed."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert schema["info"]["title"] == "EmotionLens 🎭 API"
        # Verify key endpoints are documented
        assert "/api/v1/predict" in schema["paths"]
        assert "/api/v1/predict-file" in schema["paths"]
        assert "/health" in schema["paths"]


# ── Auth Flow Integration ─────────────────────────────────────────────────


class TestAuthFlow:
    """Test API key authentication via verify_api_key function."""

    def test_open_access_when_no_key_set(self, client) -> None:
        with patch.object(api_server, "API_KEY", ""):
            result = api_server.verify_api_key(credentials=None)
            assert result is True

    def test_rejects_missing_credentials_when_key_required(self) -> None:
        with patch.object(api_server, "API_KEY", "test-secret-key"):
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                api_server.verify_api_key(credentials=None)
            assert exc_info.value.status_code == 401

    def test_rejects_wrong_api_key(self) -> None:
        with patch.object(api_server, "API_KEY", "test-secret-key"):
            from fastapi import HTTPException
            from fastapi.security import HTTPAuthorizationCredentials
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                api_server.verify_api_key(credentials=creds)
            assert exc_info.value.status_code == 403

    def test_accepts_correct_api_key(self) -> None:
        with patch.object(api_server, "API_KEY", "my-secret"):
            from fastapi.security import HTTPAuthorizationCredentials


            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="my-secret")
            result = api_server.verify_api_key(credentials=creds)
            assert result is True
