"""Tests for EmotionLens structured_logging module."""

import json
import logging
import tempfile
from pathlib import Path

from utils.structured_logging import (
    JSONFormatter,
    get_request_id,
    request_id_var,
    set_request_id,
    setup_logger,
)


class TestJSONFormatter:
    """Tests for the JSON log formatter."""

    def test_returns_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_includes_timestamp(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=1, msg="warn", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format

    def test_includes_request_id_when_set(self):
        set_request_id("test-req-123")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["request_id"] == "test-req-123"
        # Clean up
        request_id_var.set(None)

    def test_excludes_request_id_when_none(self):
        request_id_var.set(None)
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "request_id" not in data

    def test_includes_exception_info(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "boom" in data["exception"]["value"]

    def test_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="extra", args=(), exc_info=None,
        )
        record.extra_fields = {"user_id": 42, "action": "login"}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["user_id"] == 42
        assert data["action"] == "login"


class TestRequestID:
    """Tests for request ID context variable."""

    def test_set_and_get(self):
        req_id = set_request_id("my-id")
        assert req_id == "my-id"
        assert get_request_id() == "my-id"
        # Clean up
        request_id_var.set(None)

    def test_auto_generates_uuid(self):
        req_id = set_request_id()
        assert isinstance(req_id, str)
        assert len(req_id) == 12
        # Clean up
        request_id_var.set(None)

    def test_default_is_none(self):
        request_id_var.set(None)
        assert get_request_id() is None


def _cleanup_logger(logger):
    """Close and remove all handlers so temp dirs can be deleted on Windows."""
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class TestSetupLogger:
    """Tests for setup_logger factory."""

    def test_creates_logger_with_handlers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger("test_setup", log_dir=tmpdir, log_file="test.log")
            try:
                assert logger is not None
                assert logger.name == "test_setup"
                # Should have file + console handlers
                assert len(logger.handlers) >= 2
            finally:
                _cleanup_logger(logger)

    def test_returns_same_logger_on_repeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = setup_logger("test_repeat", log_dir=tmpdir)
            try:
                logger2 = setup_logger("test_repeat", log_dir=tmpdir)
                assert logger1 is logger2
            finally:
                _cleanup_logger(logger1)

    def test_logger_writes_json_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger("test_write", log_dir=tmpdir, log_file="out.jsonl")
            try:
                logger.info("test message")
                for handler in logger.handlers:
                    handler.flush()
                log_file = Path(tmpdir) / "out.jsonl"
                assert log_file.exists()
                content = log_file.read_text().strip()
                data = json.loads(content)
                assert data["message"] == "test message"
            finally:
                _cleanup_logger(logger)
