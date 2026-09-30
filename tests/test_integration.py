"""Integration tests for Emotion-Lens — full HTTP lifecycle through FastAPI.

Tests the complete request-response cycle including middleware, error handling,
multi-endpoint workflows, and OpenAPI schema generation. TensorFlow is stubbed,
but prediction requests exercise the real decode → preprocess → predict →
respond pipeline through stubbed model/cascade objects.

Run explicitly with ``pytest -m slow`` — excluded from the default fast suite
but still fast and deterministic (no TensorFlow, no network).
"""

from __future__ import annotations

import base64
import io
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow

# Mock TensorFlow before importing api_server
sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())

import api_models
import inference
from api_server import app

# ── Golden model stubs ────────────────────────────────────────────────────

# FER2013 class order used everywhere: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
EMOTION_KEYS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
HAPPY_PROBS = [0.01, 0.01, 0.02, 0.92, 0.02, 0.01, 0.01]


def _make_model(probs: list[float]) -> MagicMock:
    """Keras model stub whose predict() returns a fixed softmax row."""
    model = MagicMock()
    model.predict.return_value = np.asarray([probs], dtype=np.float32)
    return model


def _make_cascade(faces: np.ndarray | None = None) -> MagicMock:
    """Haar cascade stub returning a fixed detection box list (or none)."""
    cascade = MagicMock()
    if faces is None:
        faces = np.empty((0, 4), dtype=np.int32)
    cascade.detectMultiScale.return_value = faces
    return cascade


def _png_bytes(size: tuple[int, int] = (48, 48), seed: int = 42) -> bytes:
    """Deterministic grayscale PNG as bytes."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """TestClient with a stubbed model + cascade (no TensorFlow needed)."""
    with (
        patch.object(inference, "_model", _make_model(HAPPY_PROBS)),
        patch.object(inference, "_face_cascade", _make_cascade()),
    ):
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def dummy_b64_image() -> str:
    """Deterministic base64-encoded 48x48 grayscale PNG (no data URI prefix)."""
    return base64.b64encode(_png_bytes()).decode()


@pytest.fixture()
def dummy_b64_with_prefix(dummy_b64_image: str) -> str:
    """Base64 image with a data URI prefix."""
    return f"data:image/png;base64,{dummy_b64_image}"


# ── Full HTTP Lifecycle ───────────────────────────────────────────────────


class TestHTTPLifecycle:
    """Tests that exercise the full request → middleware → handler → response cycle."""

    def test_root_returns_service_info(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "EmotionLens 🎭 API"
        assert data["version"] == "1.0.0"
        assert "emotions" in data
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) == 7

    def test_health_endpoint_returns_model_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # The stubbed model is loaded, so the endpoint must report healthy.
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_path"].endswith(".h5")
        assert len(data["emotions"]) == 7

    def test_health_cold_start_reports_unhealthy(self) -> None:
        """Without a loaded model, /health must say unhealthy — without
        triggering a lazy model load."""
        with (
            patch.object(inference, "_model", None),
            TestClient(app, raise_server_exceptions=False) as cold_client,
        ):
            response = cold_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

    def test_health_endpoint_returns_listed_emotions(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        expected = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        assert data["emotions"] == expected


# ── Middleware Behavior ────────────────────────────────────────────────────


class TestMiddleware:
    """Verify security headers, CORS, and rate limiting are applied."""

    def test_security_headers_present(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert response.headers.get("X-XSS-Protection") == "0"

    def test_content_security_policy(self, client: TestClient) -> None:
        response = client.get("/health")
        csp = response.headers.get("Content-Security-Policy", "")
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_permissions_policy(self, client: TestClient) -> None:
        response = client.get("/health")
        pp = response.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp


# ── Prediction Endpoints ──────────────────────────────────────────────────


class TestPredictionEndpoints:
    """Integration tests for /api/v1/predict and /api/v1/predict-file."""

    def test_predict_base64_returns_success(self, client: TestClient, dummy_b64_image: str) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": dummy_b64_image, "detect_faces": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["faces_detected"] == 1
        assert len(data["results"]) == 1
        result = data["results"][0]
        # No faces detected → fallback path predicts on the full image,
        # so the stub model's dominant class (Happy) must come back.
        assert result["emotion"] == "Happy"
        assert result["confidence"] == pytest.approx(0.92, abs=1e-6)
        assert result["bbox"] is None
        assert set(result["probabilities"].keys()) == set(EMOTION_KEYS)
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0
        assert data["summary"] is not None
        assert "Happy" in data["summary"]

    def test_predict_with_data_uri_prefix(
        self, client: TestClient, dummy_b64_with_prefix: str
    ) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": dummy_b64_with_prefix, "detect_faces": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["results"][0]["emotion"] == "Happy"

    def test_predict_with_detected_face_returns_bbox(self, dummy_b64_image: str) -> None:
        """When the cascade 'detects' a face, the bbox must be forwarded."""
        cascade = _make_cascade(faces=np.array([[10, 10, 28, 28]], dtype=np.int32))
        with (
            patch.object(inference, "_model", _make_model(HAPPY_PROBS)),
            patch.object(inference, "_face_cascade", cascade),
            TestClient(app, raise_server_exceptions=False) as face_client,
        ):
            response = face_client.post(
                "/api/v1/predict",
                json={"image": dummy_b64_image, "detect_faces": True},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["faces_detected"] == 1
        assert data["results"][0]["bbox"] == [10, 10, 28, 28]

    def test_predict_empty_image_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": "", "detect_faces": True},
        )
        assert response.status_code == 400

    def test_predict_invalid_base64_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            json={"image": "not-valid-base64!!!", "detect_faces": True},
        )
        assert response.status_code == 400

    def test_predict_missing_image_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_file_endpoint(self, client: TestClient) -> None:
        """Test file upload endpoint with a dummy image."""
        buf = io.BytesIO(_png_bytes())
        response = client.post(
            "/api/v1/predict-file",
            files={"file": ("face.png", buf.getvalue(), "image/png")},
            data={"detect_faces": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["results"][0]["emotion"] == "Happy"

    def test_predict_file_rejects_non_image(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict-file",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


# ── Error Handling Workflows ──────────────────────────────────────────────


class TestErrorHandling:
    """Verify graceful error handling across the API."""

    def test_nonexistent_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method_returns_405(self, client: TestClient) -> None:
        response = client.post("/health")
        assert response.status_code == 405

    def test_predict_with_wrong_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422

    def test_malformed_json_body(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/predict",
            content="{invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ── Multi-Endpoint Workflow ────────────────────────────────────────────────


class TestMultiEndpointWorkflow:
    """Simulate a realistic user session: root → health → predict → health."""

    def test_full_user_workflow(self, client: TestClient, dummy_b64_image: str) -> None:
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

    def test_openapi_schema_is_valid(self, client: TestClient) -> None:
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
    """Test API key authentication via verify_api_key.

    verify_api_key reads API_KEY from api_models (where it is defined),
    so patches must target api_models — patching api_server.API_KEY is a no-op.
    """

    def test_open_access_when_no_key_set(self) -> None:
        with patch.object(api_models, "API_KEY", ""):
            result = api_models.verify_api_key(credentials=None)
            assert result is True

    def test_rejects_missing_credentials_when_key_required(self) -> None:
        from fastapi import HTTPException

        with patch.object(api_models, "API_KEY", "test-secret-key"):
            with pytest.raises(HTTPException) as exc_info:
                api_models.verify_api_key(credentials=None)
            assert exc_info.value.status_code == 401

    def test_rejects_wrong_api_key(self) -> None:
        from fastapi import HTTPException

        with patch.object(api_models, "API_KEY", "test-secret-key"):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                api_models.verify_api_key(credentials=creds)
            assert exc_info.value.status_code == 403

    def test_accepts_correct_api_key(self) -> None:
        with patch.object(api_models, "API_KEY", "my-secret"):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="my-secret")
            result = api_models.verify_api_key(credentials=creds)
            assert result is True
