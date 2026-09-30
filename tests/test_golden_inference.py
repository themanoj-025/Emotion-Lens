"""Golden-input tests: fixed image → expected label.

Regression net for the inference contract. The model is stubbed with a fixed
softmax row, so any change in preprocessing, class-index → label mapping, or
response shaping shows up as a deterministic failure — not a silent drift.

Covers both prediction paths that must stay in parity:
  * API path:    inference.preprocess_face → inference.predict_face
  * Camera path: utils.emotion_utils.preprocess_face → predict_emotion

TensorFlow is mocked before imports (same pattern as the other test modules).
"""

from __future__ import annotations

import base64
import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("tensorflow.keras", MagicMock())
sys.modules.setdefault("tensorflow.keras.models", MagicMock())

import inference
from api_server import app
from inference import generate_summary, predict_face, preprocess_face, process_image
from utils.emotion_utils import predict_emotion
from utils.emotion_utils import preprocess_face as preprocess_face_camera

# FER2013 class order — the single source of truth for label mapping.
EMOTION_KEYS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def _probs_for(label: str) -> list[float]:
    """One-hot-ish softmax row with 0.9 on ``label`` and 0.1/6 elsewhere."""
    weights = [0.1 / 6] * 7
    weights[EMOTION_KEYS.index(label)] = 0.9
    return weights


def _golden_face(seed: int = 2024, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Deterministic grayscale 'face' ROI (the golden input)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size, dtype=np.uint8)


def _golden_png_b64(seed: int = 2024) -> str:
    """The golden input encoded exactly as a client would send it."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_golden_face(seed), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _modelReturning(probs: list[float]) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.asarray([probs], dtype=np.float32)
    return model


# ── Label mapping (the core golden invariant) ─────────────────────────────


class TestGoldenLabelMapping:
    """Each class index must map to its FER2013 label — in both paths."""

    @pytest.mark.parametrize("label", EMOTION_KEYS)
    def test_api_path_maps_class_index_to_label(self, label: str) -> None:
        model = _modelReturning(_probs_for(label))
        emotion, confidence, probs = predict_face(model, _golden_face())

        assert emotion == label
        assert confidence == pytest.approx(0.9, abs=1e-6)
        assert list(probs.keys()) == EMOTION_KEYS
        assert probs[label] == pytest.approx(0.9, abs=1e-6)

    @pytest.mark.parametrize("label", EMOTION_KEYS)
    def test_camera_path_maps_class_index_to_label(self, label: str) -> None:
        model = _modelReturning(_probs_for(label))
        emotion, confidence, all_probs = predict_emotion(model, _golden_face())

        assert emotion == label
        assert confidence == pytest.approx(0.9, abs=1e-6)
        assert len(all_probs) == 7
        assert int(np.argmax(all_probs)) == EMOTION_KEYS.index(label)


# ── Preprocessing golden values ───────────────────────────────────────────


class TestGoldenPreprocessing:
    """Both preprocessing paths must produce identical model-ready tensors."""

    def test_output_shape_and_range(self) -> None:
        tensor = preprocess_face(_golden_face())
        assert tensor.shape == (1, 48, 48, 1)
        assert tensor.dtype == np.float32
        assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0

    def test_api_and_camera_paths_are_bit_identical(self) -> None:
        """Camera parity: same ROI in → identical tensor out, either path."""
        roi = _golden_face()
        api_tensor = preprocess_face(roi)
        camera_tensor = preprocess_face_camera(roi)
        np.testing.assert_array_equal(api_tensor, camera_tensor)

    def test_known_pixel_values_survive_roundtrip(self) -> None:
        """Downscale of a constant image must preserve its exact value."""
        roi = np.full((96, 96), 128, dtype=np.uint8)
        tensor = preprocess_face(roi)
        assert float(tensor.mean()) == pytest.approx(128 / 255.0, abs=1e-6)

    def test_deterministic_across_calls(self) -> None:
        roi = _golden_face()
        first = predict_face(_modelReturning(_probs_for("Happy")), roi)
        second = predict_face(_modelReturning(_probs_for("Happy")), roi)
        assert first == second


# ── Golden image → response (process_image + summary) ─────────────────────


class TestGoldenImageToResponse:
    """Full decode → detect → predict → summarize, with a stubbed model."""

    def _process(self, model: MagicMock, detect_faces: bool):
        # Grayscale → BGR is just 3 stacked channels; build it with numpy so
        # this test can't pick up the cv2 MagicMock that other test modules
        # plant in sys.modules during collection.
        gray = _golden_face()
        img_bgr = np.stack([gray] * 3, axis=-1)
        cascade = MagicMock()
        cascade.detectMultiScale.return_value = np.empty((0, 4), dtype=np.int32)
        return process_image(model, cascade, img_bgr, detect_faces)

    def test_fixed_image_yields_expected_label(self) -> None:
        model = _modelReturning(_probs_for("Surprise"))
        results, count = self._process(model, detect_faces=True)

        assert count == 1
        assert len(results) == 1
        assert results[0].emotion == "Surprise"
        assert results[0].confidence == pytest.approx(0.9, abs=1e-6)
        assert results[0].bbox is None  # no detection → full-image fallback

    def test_summary_matches_result(self) -> None:
        model = _modelReturning(_probs_for("Sad"))
        results, _ = self._process(model, detect_faces=True)
        assert generate_summary(results) == "Detected: Sad (90.0%)"


class TestGoldenAPIContract:
    """HTTP contract: POST the golden image → exact JSON payload."""

    def test_predict_returns_exact_golden_payload(self) -> None:
        with (
            patch.object(inference, "_model", _modelReturning(_probs_for("Happy"))),
            patch.object(inference, "_face_cascade", MagicMock()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(
                "/api/v1/predict",
                json={"image": _golden_png_b64(), "detect_faces": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["faces_detected"] == 1
        result = data["results"][0]
        assert result["emotion"] == "Happy"
        assert result["confidence"] == pytest.approx(0.9, abs=1e-6)
        assert result["probabilities"] == {
            label: pytest.approx(p, abs=1e-6) for label, p in zip(EMOTION_KEYS, _probs_for("Happy"))
        }
        assert data["summary"] == "Detected: Happy (90.0%)"

    def test_repeated_posts_are_deterministic(self) -> None:
        payload = {"image": _golden_png_b64(), "detect_faces": True}
        with (
            patch.object(inference, "_model", _modelReturning(_probs_for("Neutral"))),
            patch.object(inference, "_face_cascade", MagicMock()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            first = client.post("/api/v1/predict", json=payload).json()
            second = client.post("/api/v1/predict", json=payload).json()

        assert first["results"] == second["results"]
        assert first["summary"] == second["summary"]
