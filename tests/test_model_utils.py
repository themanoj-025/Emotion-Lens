import pytest

pytestmark = pytest.mark.unit

"""Tests for model utility functions."""
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before import (PIL NOT mocked — needed by other tests)
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()
# streamlit isn't in the CI test job's deps — mock it if genuinely missing.
# cv2 must NOT be unconditionally mocked: a planted MagicMock leaks into any
# test that lazily imports cv2 at call time (e.g. /predict-file → 500s), so
# only mock it when cv2 truly isn't importable in this environment. Use
# importlib so the probing import itself doesn't trip unused-import lint.
try:
    importlib.import_module("cv2")
except ImportError:
    sys.modules["cv2"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

from utils.model_utils import (
    EMOTION_CONFIG,
    EMOTIONS,
    MODEL_PATH,
    MOOD_MUSIC_MAP,
    get_model_summary,
    is_model_available,
    load_face_cascade,
)


class TestEmotions:
    """Tests for emotion constants and configuration."""

    def test_emotions_is_list_of_7(self) -> None:
        assert isinstance(EMOTIONS, list)
        assert len(EMOTIONS) == 7

    def test_emotions_are_strings(self) -> None:
        for e in EMOTIONS:
            assert isinstance(e, str)

    def test_emotion_config_has_all_emotions(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG

    def test_emotion_config_keys_have_required_fields(self) -> None:
        for emotion, config in EMOTION_CONFIG.items():
            assert "color" in config
            assert "emoji" in config
            assert "bg" in config


class TestModelConstants:
    """Tests for module-level constants."""

    def test_model_path_is_string(self) -> None:
        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".h5")

    def test_emotions_colors_are_hex(self) -> None:
        for emotion, config in EMOTION_CONFIG.items():
            assert config["color"].startswith("#")
            assert len(config["color"]) == 7


class TestMoodMusicMap:
    """Tests for MOOD_MUSIC_MAP — every emotion needs playable links."""

    def test_all_emotions_have_music(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in MOOD_MUSIC_MAP

    def test_music_has_spotify_and_youtube(self) -> None:
        for emotion in EMOTIONS:
            music = MOOD_MUSIC_MAP[emotion]
            assert "spotify" in music
            assert "youtube" in music


class TestLoadFaceCascade:
    """Tests for face cascade loading."""

    def test_returns_callable(self) -> None:
        result = load_face_cascade()
        assert result is not None


class TestIsModelAvailable:
    """Tests for is_model_available function."""

    def test_returns_false_when_no_model(self) -> None:
        with patch("os.path.exists", return_value=False):
            assert is_model_available() is False


class TestGetModelSummary:
    """Tests for get_model_summary function."""

    def test_returns_tuple(self) -> None:
        mock_model = MagicMock()
        mock_model.layers = []
        mock_model.count_params.return_value = 1000000
        result = get_model_summary(mock_model)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_layers_and_params(self) -> None:
        mock_layer = MagicMock()
        mock_layer.name = "dense"
        mock_layer.__class__ = type("Dense", (), {})
        mock_layer.output_shape = (None, 7)
        mock_layer.count_params.return_value = 50000
        mock_layer.trainable = True

        mock_model = MagicMock()
        mock_model.layers = [mock_layer]
        mock_model.count_params.return_value = 50000
        layers_info, params = get_model_summary(mock_model)
        assert isinstance(layers_info, list)
        assert isinstance(params, dict)
        assert "total" in params


class TestModelUtilsIntegration:
    """Integration tests for model utilities."""

    def test_emotion_config_complete(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG
            config = EMOTION_CONFIG[emotion]
            assert config["color"].startswith("#")
            assert len(config["emoji"]) > 0

    def test_model_path_ends_with_h5(self) -> None:
        assert MODEL_PATH.endswith(".h5")

    def test_emotion_list_matches_config(self) -> None:
        assert set(EMOTIONS) == set(EMOTION_CONFIG.keys())


class TestTryDownloadModel:
    """Retry/timeout hardening for the cloud boot download (no real network)."""

    PAYLOAD = b"x" * 150_000  # > _MIN_MODEL_BYTES so validation passes

    @staticmethod
    def _fake_requests(monkeypatch, get_impl, exc_class=None):
        import types

        fake = types.ModuleType("requests")
        fake.RequestException = exc_class or type("RequestException", (Exception,), {})
        fake.get = get_impl
        monkeypatch.setitem(sys.modules, "requests", fake)
        return fake

    @staticmethod
    def _setup(monkeypatch, tmp_path):
        import utils.model_utils as mu

        model_path = str(tmp_path / "emotion_model.h5")
        monkeypatch.setattr(mu, "MODEL_PATH", model_path)
        monkeypatch.setattr(mu, "MODEL_DOWNLOAD_URLS", ["https://example.com/model.h5"])
        monkeypatch.setattr(mu, "is_model_available", lambda: os.path.exists(model_path))
        sleeps: list[float] = []

        def _record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(mu.time, "sleep", _record_sleep)
        return model_path, sleeps

    def test_success_first_attempt(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        model_path, sleeps = self._setup(monkeypatch, tmp_path)
        timeouts_seen = []

        def get_impl(url, **kwargs):
            timeouts_seen.append(kwargs.get("timeout"))
            return _StreamingResponse(200, self.PAYLOAD)

        self._fake_requests(monkeypatch, get_impl)
        assert mu.try_download_model() is True
        assert os.path.exists(model_path)
        with open(model_path, "rb") as f:
            assert f.read() == self.PAYLOAD
        assert not os.path.exists(model_path + ".part")  # temp cleaned up
        assert sleeps == []  # no retries needed
        # Connect/read timeout tuple must be passed to requests.get
        assert timeouts_seen and timeouts_seen[0] == (
            mu._DOWNLOAD_CONNECT_TIMEOUT,
            mu._DOWNLOAD_READ_TIMEOUT,
        )

    def test_transient_503_then_success(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        model_path, sleeps = self._setup(monkeypatch, tmp_path)
        responses = iter(
            [
                _StreamingResponse(503, b""),
                _StreamingResponse(503, b""),
                _StreamingResponse(200, self.PAYLOAD),
            ]
        )
        self._fake_requests(monkeypatch, lambda url, **kw: next(responses))

        assert mu.try_download_model() is True
        assert os.path.exists(model_path)
        assert len(sleeps) == 2  # backed off between the failed attempts

    def test_permanent_404_fails_fast(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        model_path, sleeps = self._setup(monkeypatch, tmp_path)
        calls: list[str] = []

        def get_impl(url: str, **kw):
            calls.append(url)
            return _StreamingResponse(404, b"")

        self._fake_requests(monkeypatch, get_impl)

        assert mu.try_download_model() is False
        assert len(calls) == 1  # no retries for a permanent failure
        assert sleeps == []
        assert not os.path.exists(model_path)

    def test_network_errors_retry_then_give_up(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        model_path, sleeps = self._setup(monkeypatch, tmp_path)
        attempts = []

        def get_impl(url, **kwargs):
            attempts.append(url)
            raise mu_requests_error()

        def mu_requests_error():
            raise requests_exc

        fake = self._fake_requests(monkeypatch, get_impl)
        requests_exc = fake.RequestException("connection reset")

        assert mu.try_download_model() is False
        assert len(attempts) == 3  # bounded retries
        assert len(sleeps) == 2  # backoff between attempts
        assert not os.path.exists(model_path)
        assert not os.path.exists(model_path + ".part")  # no partial file left

    def test_all_urls_exhausted_returns_false(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        model_path, _ = self._setup(monkeypatch, tmp_path)
        monkeypatch.setattr(
            mu,
            "MODEL_DOWNLOAD_URLS",
            ["https://a.example/m.h5", "https://b.example/m.h5"],
        )
        self._fake_requests(
            monkeypatch,
            lambda url, **kw: _StreamingResponse(404, b""),
        )

        assert mu.try_download_model() is False
        assert not os.path.exists(model_path)

    def test_already_available_skips_download(self, monkeypatch, tmp_path) -> None:
        import utils.model_utils as mu

        monkeypatch.setattr(mu, "is_model_available", lambda: True)
        called = []
        self._fake_requests(monkeypatch, lambda url, **kw: called.append(url))

        assert mu.try_download_model() is True
        assert called == []

    def test_oversized_but_invalid_file_is_retried_and_cleaned(self, monkeypatch, tmp_path) -> None:
        """A payload below _MIN_MODEL_BYTES must not be published."""
        import utils.model_utils as mu

        model_path, sleeps = self._setup(monkeypatch, tmp_path)
        responses = iter([_StreamingResponse(200, b"tiny")] * 3)
        self._fake_requests(monkeypatch, lambda url, **kw: next(responses))

        assert mu.try_download_model() is False
        assert not os.path.exists(model_path)  # corrupt file never published
        assert not os.path.exists(model_path + ".part")
        assert len(sleeps) == 2  # retried on each failed validation


class _StreamingResponse:
    """Minimal requests.Response stub for stream=True downloads."""

    def __init__(self, status_code: int, payload: bytes) -> None:
        self.status_code = status_code
        self._payload = payload

    def iter_content(self, chunk_size: int):
        if self._payload:
            yield self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False
