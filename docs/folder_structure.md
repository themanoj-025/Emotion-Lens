# EmotionLens — Folder Structure

```
Emotion-Lens/
├── streamlit_app.py              # Streamlit entry (multi-page)
├── api_server.py                 # FastAPI REST server
├── train.py                      # Training CLI
├── inference.py                  # Single-image inference CLI
├── webcam_inference.py           # Real-time webcam inference CLI
│
├── utils/                        # Shared helpers
│   ├── config.py                 # Paths & constants
│   ├── model_utils.py            # Model load / availability
│   ├── emotion_utils.py          # Emotion labels & post-processing
│   ├── session_utils.py          # Streamlit session state
│   ├── chart_utils.py            # Analytics charts
│   ├── export_utils.py           # Export helpers
│   ├── gradcam_utils.py          # GradCAM visualizations
│   └── smoothing_utils.py        # Prediction smoothing
│
├── pages/                        # Streamlit pages
│   ├── page1_live_camera.py
│   ├── page2_image_analysis.py
│   ├── page3_analytics.py
│   ├── page4_train_model.py
│   ├── page5_model_inspector.py
│   ├── page6_emotion_game.py
│   └── page7_about.py
│
├── assets/                       # style.css
├── .streamlit/                   # config.toml
├── .devcontainer/                # Dev container
├── .vscode/                      # IDE settings
├── docs/                         # Full documentation suite
│   ├── architecture.md · folder_structure.md · module_dependency.md
│   ├── startup_flow.md · package_overview.md
│   ├── migration/                # migration_summary, old_tree_to_new_tree, file_move_ledger
│   ├── project/                  # analysis_report.md, plans, tracker
│   ├── community/ design/ product/ reference/ technical/
├── .github/                      # CI, CodeQL, gitleaks, automation workflows
├── Dockerfile
├── docker-compose.yml / .dev.yml / .prod.yml
├── pyproject.toml                # Tooling config (black, isort)
├── requirements.txt              # Python deps
├── packages.txt                  # System deps
├── runtime.txt                   # Python version
├── Makefile                      # Docker task runner
├── .env.example                  # Env-var template
├── .editorconfig / .gitattributes / .gitignore
├── AGENTS.md                     # Agent instructions
└── README.md, LICENSE, PROJECT_ANALYSIS.md, PROJECT_OVERVIEW.md
```

## Root Hygiene

- Root holds entry points + manifests + top-level dirs only.
- `AGENTS_FIX.md` (AI-scaffolding duplicate) **removed** in this pass; stale references in `.dockerignore` and `PROJECT_OVERVIEW.md` cleaned.
- `saved_model/` (trained artifacts) correctly gitignored — not committed.

## v6 Restructure (2026-08-11)

| Old path | New path | Reason | Mechanism |
|---|---|---|---|
| `docs/migration_summary.md` | `docs/migration/migration_summary.md` | Consolidate migration records under `docs/migration/` (Principal Architect protocol Phase 6) | `git mv` |
| — | `docs/module_dependency.md` | Phase 6 deliverable | added |
| — | `docs/startup_flow.md` | Phase 6 deliverable | added |
| — | `docs/package_overview.md` | Phase 6 deliverable | added |
| — | `docs/migration/old_tree_to_new_tree.md` | Phase 6 deliverable | added |
| — | `docs/migration/file_move_ledger.md` | Phase 6 deliverable | added |

No application files moved: the flat layout is the canonical Streamlit multipage
pattern (entry script + sibling `pages/` + importable `utils/` package), wired into
Docker, CI (pyflakes + import checks), and Streamlit Cloud — see
`docs/migration/file_move_ledger.md` for the contract analysis.
