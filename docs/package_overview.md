# Package Overview — Emotion-Lens

Inventory of every module (post-restructure). The app is a flat-layout TensorFlow +
Streamlit application with four standalone entry points, seven Streamlit pages, and a
shared `utils/` package.

## 1. Entry Points (root)

| Module | Responsibility | Entry point |
| --- | --- | --- |
| `streamlit_app.py` | Main multipage Streamlit dashboard (live camera, image analysis, analytics, training UI, model inspector, emotion game, about). | `streamlit run streamlit_app.py` |
| `api_server.py` | FastAPI REST server: `GET /health`, `POST /predict`, `POST /predict-file`, `GET /`. | `uvicorn api_server:app` |
| `train.py` | CNN training CLI (tensorflow + kagglehub). | `python train.py` |
| `inference.py` | Single-image emotion inference CLI. | `python inference.py` |
| `webcam_inference.py` | Real-time webcam emotion detection CLI (with smoothing). | `python webcam_inference.py` |

## 2. Shared Package (`utils/`)

| Module | Responsibility |
| --- | --- |
| `utils/config.py` | Paths, constants, plotly layout config. |
| `utils/model_utils.py` | Model load/availability, cloud artifact ensure, prediction API. |
| `utils/emotion_utils.py` | Emotion labels, post-processing, positivity score, summaries. |
| `utils/session_utils.py` | Streamlit session-state init/persistence helpers. |
| `utils/chart_utils.py` | Analytics charts (bar/radar/pie). |
| `utils/export_utils.py` | CSV/export helpers for predictions. |
| `utils/gradcam_utils.py` | GradCAM heatmap generation + overlay. |
| `utils/smoothing_utils.py` | Temporal prediction smoothing. |

## 3. Pages (`pages/`)

| Module | Responsibility |
| --- | --- |
| `page1_live_camera.py` | Live webcam emotion feed. |
| `page2_image_analysis.py` | Single-image upload + analysis. |
| `page3_analytics.py` | Session analytics dashboards. |
| `page4_train_model.py` | In-app training UI. |
| `page5_model_inspector.py` | Model architecture/metrics inspector. |
| `page6_emotion_game.py` | Interactive emotion game. |
| `page7_about.py` | About/help. |

## 4. Assets & Config

| Path | Responsibility |
| --- | --- |
| `assets/style.css`, `assets/dashboard-preview.svg` | UI styling + preview asset. |
| `.streamlit/config.toml` | Streamlit runtime config. |
| `.devcontainer/devcontainer.json` | Dev container definition. |
| `packages.txt` / `runtime.txt` | System deps + Python version pin (Docker/Cloud). |
| `saved_model/` | Trained artifacts — **gitignored**, never committed. |

## 5. Infrastructure

`Dockerfile` (multi-stage), `docker-compose.yml`/`.dev.yml`/`.prod.yml`, `Makefile`
(compose ergonomics), `.github/workflows/` (ci, codeql, gitleaks, labeler,
maintenance, stale, welcome).

## 6. Documentation (`docs/`)

Root suite: `architecture.md`, `folder_structure.md`, `module_dependency.md`,
`startup_flow.md`, `package_overview.md`. Migration records: `migration/`
(`migration_summary.md` ← v5.0, `old_tree_to_new_tree.md`, `file_move_ledger.md`).
Categorized docs: `community/`, `design/`, `product/`, `project/`, `reference/`,
`technical/`.

## 7. Test Coverage

**No pytest suite exists.** CI enforces syntax (py_compile), static analysis
(pyflakes), per-module import checks, Bandit, and Docker build + Trivy. A unit-test
follow-up for the pure `utils/` modules (`emotion_utils`, `smoothing_utils`,
`session_utils`, `config`) is recommended (see file-move ledger / backlog).
