# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════════
# Emotion-Lens — Streamlit emotion recognition app
#
# Build targets:
#   prod (default) — production Streamlit server (:8501)
#   dev            — hot reload for local development
#
# The trained model checkpoint is NOT baked in — it is downloaded at
# first run via kagglehub (see requirements.txt). Runtime needs network
# access on first prediction.
#
# Usage:
#   docker build -t emotion-lens .
#   docker compose up -d
# ═══════════════════════════════════════════════════════════════════════

# ── Base stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Emotion-Lens"
LABEL org.opencontainers.image.description="Streamlit emotion recognition from images/webcam"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Emotion-Lens"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# OpenCV/ML runtime libs (existing deps) + tini (PID 1) + curl (healthcheck)
RUN apt-get update && apt-get install -y \
        tini \
        curl \
        libgl1 libglib2.0-0 libsm6 \
        libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Deps stage ─────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Prod stage ─────────────────────────────────────────────────────────
FROM deps AS prod

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

# Application source — includes the pages/ and utils/ packages the app
# imports at runtime.
COPY streamlit_app.py inference.py api_server.py train.py webcam_inference.py ./
COPY pages/ ./pages/
COPY utils/ ./utils/
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ── Dev stage: hot reload ──────────────────────────────────────────────
FROM deps AS dev

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

COPY streamlit_app.py inference.py api_server.py train.py webcam_inference.py ./
COPY pages/ ./pages/
COPY utils/ ./utils/
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.fileWatcherType=polling", \
     "--server.runOnSave=true"]
