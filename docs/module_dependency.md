# Module Dependency — Emotion-Lens

**No circular imports.** The graph is a star: four standalone entry points and seven
Streamlit pages all depend on the shared `utils/` leaf modules; entry points do not
import each other.

## 1. Dependency Graph

```
  ENTRY POINTS (standalone, no cross-imports)
  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────────┐
  │streamlit_app │ │api_server  │ │train.py  │ │webcam_inference  │
  │(main UI)     │ │(FastAPI)   │ │(training)│ │(webcam CLI)      │
  └──────┬───────┘ └──────┬─────┘ └──────────┘ └──────────────────┘
         │                │
         ▼                ▼
  ┌────────────────────────────────────────────┐
  │ pages/page1..page7 (Streamlit multipage)   │
  │  — discovered by Streamlit beside the      │
  │    entry script (framework convention)     │
  └──────────────────┬─────────────────────────┘
                     │  from utils.<module> import ...
                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ utils/  (shared leaf package — imported as top-level pkg)    │
  │  config.py · model_utils.py · emotion_utils.py               │
  │  session_utils.py · chart_utils.py · export_utils.py         │
  │  gradcam_utils.py · smoothing_utils.py                       │
  └──────────────────────────────────────────────────────────────┘
```

## 2. Dependency Matrix

| Module | Imports | Depends on | Consumed by |
| --- | --- | --- | --- |
| `streamlit_app.py` | `utils.session_utils`, `utils.model_utils` | streamlit, tensorflow (via model_utils) | `streamlit run streamlit_app.py` (Docker, Cloud) |
| `pages/page1..7` | `utils.emotion_utils`, `utils.model_utils`, `utils.session_utils`, `utils.chart_utils`, ... | streamlit, tensorflow | Streamlit multipage discovery |
| `api_server.py` | cv2, numpy, PIL, tensorflow, fastapi | — | `uvicorn api_server:app` (standalone REST) |
| `train.py` | tensorflow, kagglehub, argparse | — | `python train.py` (training CLI) |
| `inference.py` | cv2, numpy, tensorflow | — | `python inference.py` (single-image CLI) |
| `webcam_inference.py` | cv2, numpy, tensorflow | — | `python webcam_inference.py` (real-time CLI) |
| `utils/config.py` | — (leaf, constants/paths) | — | all utils + pages |
| `utils/model_utils.py` | `utils.config` | tensorflow | streamlit_app, pages, api path |
| `utils/emotion_utils.py` | `utils.config` | numpy | pages, api |
| `utils/session_utils.py` | `utils.config` | streamlit | streamlit_app, pages |
| `utils/chart_utils.py` | `utils.config` | plotly | pages |
| `utils/export_utils.py` | pandas | — | pages |
| `utils/gradcam_utils.py` | tensorflow, cv2 | — | pages |
| `utils/smoothing_utils.py` | numpy | — | pages |

## 3. Why This Shape (contract analysis)

- **`pages/` must be a sibling of the entry script** — Streamlit discovers multipage
  routes from a `pages/` folder next to `streamlit_app.py`. Relocating `streamlit_app.py`
  into `app/` would require `app/pages/` and would break the Docker COPY, Streamlit
  Cloud entry config, and CI's pyflakes/import matrix — with zero functional gain.
- **`utils/` is an importable top-level package** (`from utils.xxx import ...`) used by
  CI's verification steps and every page. Moving it would rewrite ~20 import
  statements + ~40 CI references for no benefit.
- **Entry points are standalone leaves** — no mutual imports, so each is independently
  runnable and independently testable.

## 4. Change Warnings

- Renaming/moving `pages/` or `utils/` breaks Streamlit discovery and CI import checks
  (`ci.yml` has per-module `from utils.* import ...` verification steps).
- `model_utils.py` and `gradcam_utils.py` import TensorFlow — keep them separate from
  pure-logic modules so CI's no-model checks stay runnable.
- Adding a page requires no registration (auto-discovery) — only `pages/pageN_*.py`.
