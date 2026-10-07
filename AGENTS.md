# AGENTS.md — Emotion-Lens

> Canonical project instructions. Pointers like `CLAUDE.md` or
> `.github/copilot-instructions.md` should say "See AGENTS.md".

---

## Project overview

**Emotion-Lens** — a textual emotion detection and analysis platform. Core
components:

- **API** — request/response service for emotion classification.
- **Model** — transformer-based emotion classifier fine-tuned on a
  multi-label emotion dataset.
- **Dashboard** — Streamlit app for exploring predictions and dataset
  stats.
- **NLP** — preprocessing, tokenization, and dataset loaders.

Stack: Python 3.11+ · FastAPI · Hugging Face Transformers · PyTorch ·
Streamlit.

---

## Exact commands

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Lint / typecheck / test
make lint
pre-commit run --all-files
python -m mypy . --ignore-missing-imports
python -m pytest tests/ -v --cov=. --cov-fail-under=70

# Run
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

---

## Folder map

| Path | Purpose |
|------|---------|
| `api/` | FastAPI application (routes, services) |
| `model/` | Model definition, training, inference |
| `dashboard/` | Streamlit dashboard |
| `nlp/` | Preprocessing, tokenization, datasets |
| `tests/` | pytest suite |
| `.github/workflows/` | CI (ruff, mypy, pytest, gitleaks, trivy) |

## Do / don't

- **Do** keep the model interface decoupled from the API so the backend
  can swap transformers for a faster variant.
- **Do not** commit `.env` files.
- **Do not** commit model weights with a `.git` hook that strips them.

## Security rules

- No secrets in the repository; `gitleaks` CI gate gates on hits.
- Rate-limit the emotion API to avoid abuse.

## AI-assistance convention

Commits authored by AI must carry the trailer:

```text
AI-Assisted: yes | no | partial
```

See `.gitmessage` for the template. Do not rewrite historic commits
retroactively.
