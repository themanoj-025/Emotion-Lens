# Startup Flow — Emotion-Lens

## 1. Streamlit App Boot (primary surface)

```
streamlit run streamlit_app.py          # local / Docker CMD / Streamlit Cloud
│
├─ 1. streamlit_app.py imports utils.session_utils + utils.model_utils
├─ 2. init_session_state() — session defaults (history, filters, game state)
├─ 3. Main page renders; Streamlit auto-discovers pages/page1..page7
│      (multipage navigation sidebar)
├─ 4. On demand: model loaded via utils.model_utils (cached) —
│      ensure_model_on_cloud() pulls the artifact if absent
└─ 5. Server ready on :8501 (healthcheck curl /_stcore/health)
```

## 2. FastAPI Server Boot (secondary surface)

```
python api_server.py        # or: uvicorn api_server:app --port 8000
│
├─ 1. FastAPI app constructed with /health, /predict, /predict-file, /
├─ 2. Model loaded lazily on first prediction
└─ 3. Ready on :8000
```

## 3. CLI Surfaces

| Command | Purpose |
| --- | --- |
| `python train.py` | Train the CNN (tensorflow + kagglehub dataset). |
| `python inference.py <image>` | Single-image emotion prediction. |
| `python webcam_inference.py` | Real-time webcam emotion detection with smoothing. |

## 4. Docker Boot

- **Prod (default)**: `python:3.11-slim` base → deps → `COPY streamlit_app.py
  inference.py api_server.py train.py webcam_inference.py .` + `COPY pages/ utils/
  assets/ .streamlit/` → `CMD ["streamlit", "run", "streamlit_app.py", ...]`;
  healthcheck on `/_stcore/health`.
- **Dev target**: same COPY set + `--server.fileWatcherType=polling
  --server.runOnSave=true` for hot reload (`docker-compose.dev.yml` bind mounts).
- **Makefile**: `up/build/shell/health/config/reset` compose ergonomics.

## 5. CI (push/PR)

`ci.yml` runs: pyflakes over root entry points + `utils/` + `pages/`; a long matrix of
per-module import-verification steps (`from utils.config import ...`,
`from utils.model_utils import ...`, ...); `py_compile` over the full file matrix;
Bandit; lychee; Docker build + Trivy. **No pytest suite exists** — CI is
syntax/import/security/build gated.

## 6. Environment

`.env.example` documents required vars (model paths, upload limits). `runtime.txt`
pins Python for Streamlit Cloud. `packages.txt` lists system packages for the Docker
build.

## 7. Failure Modes

| Failure | Behavior |
| --- | --- |
| Model artifact missing | `utils.model_utils.ensure_model_on_cloud()` attempts download; UI shows guidance if unavailable |
| Upload too large | `api_server.py` upload-size limits (flagged in v5.0 summary) |
| No webcam | `webcam_inference.py` exits with a clear error |
| CI import of tensorflow modules | Fails loudly on missing deps (CI installs requirements) |
