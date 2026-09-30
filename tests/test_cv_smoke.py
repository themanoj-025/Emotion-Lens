"""Smoke tests: real OpenCV → prediction pipeline (item: lock OpenCV & verify).

Guards against the OpenCV 5.0 C-API breakage (5.x wheels dropped the bundled
haarcascade data and moved ``CascadeClassifier``), and proves the minimal
image → preprocessing → prediction chain works with the *real* cv2 — not the
``sys.modules["cv2"] = MagicMock()`` stub other test modules plant.

Keep this module import-light and fast (<1s).
"""

from __future__ import annotations

import base64
import importlib
import io
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Pinned in requirements.txt; OpenCV 5.x wheels drop haarcascade data.
_EXPECTED_OPENCV_PIN = r"opencv-(?:contrib-)?python-headless==4\.14\.0\.94"
_EXPECTED_CV_VERSION_MAJOR = 4

_real_cv2_cache: types.ModuleType | None = None


def _real_cv2() -> types.ModuleType:
    """Return the real cv2 module even if another test module planted a mock."""
    global _real_cv2_cache
    if _real_cv2_cache is not None:
        return _real_cv2_cache

    # A MagicMock may have been planted in sys.modules (e.g. test_model_utils)
    # or cv2 may be absent entirely — either way, load the genuine package
    # out-of-band, then restore the previous state so later test modules are
    # unaffected.
    existing = sys.modules.pop("cv2", None)
    real = importlib.import_module("cv2")
    if existing is not None:
        sys.modules["cv2"] = existing  # restore planted mock, if any
    _real_cv2_cache = real
    return real


def _synthetic_face_bgr(size: tuple[int, int] = (120, 120)) -> np.ndarray:
    """Deterministic synthetic grayscale 'face' with facial-feature contrast."""
    rng = np.random.default_rng(7)
    gray = rng.integers(60, 200, size, dtype=np.uint8)
    h, w = size
    # Dark eye band and mouth band on a lighter face — enough structure for
    # the Haar cascade smoke check and realistic preprocessing.
    gray[h // 4 : h // 3, w // 5 : w - w // 5] = 30
    gray[2 * h // 3 : 3 * h // 4, w // 3 : w - w // 3] = 45
    return np.stack([gray] * 3, axis=-1)  # gray → BGR


# ── Version pin enforcement ───────────────────────────────────────────────


class TestOpenCVPin:
    def test_requirements_pins_compatible_opencv(self) -> None:
        requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
            encoding="utf-8"
        )
        assert re.search(_EXPECTED_OPENCV_PIN, requirements), (
            "requirements.txt must pin a 4.x opencv build — 5.x wheels drop "
            "the bundled haarcascade data this app depends on"
        )

    def test_runtime_opencv_is_4x(self) -> None:
        cv2 = _real_cv2()
        major = int(cv2.__version__.split(".")[0])
        assert major == _EXPECTED_CV_VERSION_MAJOR, (
            f"OpenCV {cv2.__version__} installed; this app requires 4.x "
            "(5.x removed CascadeClassifier C-API data)"
        )


# ── Real OpenCV capability checks ─────────────────────────────────────────


class TestOpenCVSmoke:
    def test_haarcascade_data_is_bundled(self) -> None:
        cv2 = _real_cv2()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        assert Path(cascade_path).exists(), (
            f"haarcascade data missing at {cascade_path} — OpenCV wheel lost "
            "its bundled cascades (the 5.0 breakage this pin guards against)"
        )

    def test_cascade_loads_non_empty(self) -> None:
        cv2 = _real_cv2()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        assert not cascade.empty()

    def test_inter_area_resize_matches_inference_path(self) -> None:
        cv2 = _real_cv2()
        img = _synthetic_face_bgr()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        assert resized.shape == (48, 48)
        assert resized.dtype == np.uint8


# ── Image → prediction chain (real cv2 + stubbed model) ───────────────────


class TestImageToPredictionSmoke:
    """End-to-end smoke: PNG bytes → decode → preprocess → label."""

    def _encode_png_b64(self) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(_synthetic_face_bgr()).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_decode_base64_to_bgr(self) -> None:
        inference = importlib.import_module("inference")
        img_bgr = inference.decode_base64_image(self._encode_png_b64())
        assert img_bgr.ndim == 3
        assert img_bgr.shape[2] == 3
        assert img_bgr.shape[:2] == (120, 120)

    def test_preprocess_then_predict_yields_label(self) -> None:
        cv2 = _real_cv2()
        inference = importlib.import_module("inference")
        from api_models import EMOTIONS

        img_bgr = inference.decode_base64_image(self._encode_png_b64())
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Stub the model with a fixed softmax row (Sad dominant, index 5).
        probs = np.full(7, 0.1 / 6, dtype=np.float32)
        probs[EMOTIONS.index("Sad")] = 0.9

        class _StubModel:
            def predict(self, batch, **_kwargs):
                assert batch.shape == (1, 48, 48, 1)
                assert batch.dtype == np.float32
                return np.asarray([probs])

        emotion, confidence, prob_dict = inference.predict_face(_StubModel(), gray)
        assert emotion == "Sad"
        assert confidence == pytest.approx(0.9, abs=1e-6)
        assert set(prob_dict) == set(EMOTIONS)

    def test_process_image_end_to_end(self) -> None:
        cv2 = _real_cv2()
        inference = importlib.import_module("inference")
        from api_models import EMOTIONS

        probs = np.full(7, 0.1 / 6, dtype=np.float32)
        probs[EMOTIONS.index("Happy")] = 0.9

        class _StubModel:
            def predict(self, batch, **_kwargs):
                return np.asarray([probs])

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        img_bgr = inference.decode_base64_image(self._encode_png_b64())

        results, count = inference.process_image(_StubModel(), cascade, img_bgr, detect_faces=True)
        # Synthetic pattern may or may not trip the Haar detector; both paths
        # must return exactly one well-formed prediction.
        assert count == 1
        assert len(results) == 1
        assert results[0].emotion in EMOTIONS
        assert 0.0 <= results[0].confidence <= 1.0
