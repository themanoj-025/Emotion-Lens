# Old Tree → New Tree — Emotion-Lens

Restructure performed **2026-08-11** (v6, Principal Architect protocol). The flat
layout is the canonical Streamlit multipage pattern and was retained (see
`docs/module_dependency.md` §3 for the contract analysis). **Zero code changes, zero
import changes, zero entry-point changes.**

## Before (2026-08-10, after v5.0)

```
Emotion-Lens/
├── streamlit_app.py · api_server.py · train.py · inference.py · webcam_inference.py
├── pages/page1..page7
├── utils/ (8 modules)
├── assets/ · .streamlit/ · .devcontainer/ · .vscode/
├── docs/
│   ├── architecture.md · folder_structure.md · migration_summary.md   ← root of docs/
│   ├── community/ design/ product/ project/ reference/ technical/
├── .github/workflows/ (7 workflows)
├── Dockerfile · docker-compose*.yml · Makefile
├── pyproject.toml · requirements.txt · packages.txt · runtime.txt · .env.example
├── README.md · PROJECT_OVERVIEW.md · PROJECT_ANALYSIS.md · AGENTS.md
└── .gitignore · .dockerignore · .editorconfig · .gitattributes
```

## After (2026-08-11)

```
Emotion-Lens/
├── streamlit_app.py · api_server.py · train.py · inference.py · webcam_inference.py   (unchanged)
├── pages/ · utils/ · assets/ · .streamlit/ · .devcontainer/ · .vscode/                (unchanged)
├── docs/
│   ├── architecture.md · folder_structure.md                    (existing, kept + updated)
│   ├── module_dependency.md                                     (NEW)
│   ├── startup_flow.md                                          (NEW)
│   ├── package_overview.md                                      (NEW)
│   ├── migration/
│   │   ├── migration_summary.md                                 (MOVED from docs/)
│   │   ├── old_tree_to_new_tree.md                              (NEW — this file)
│   │   └── file_move_ledger.md                                  (NEW)
│   ├── community/ design/ product/ project/ reference/ technical/ (unchanged)
├── .github/workflows/                                           (unchanged)
├── Dockerfile · docker-compose*.yml · Makefile                  (unchanged)
├── pyproject.toml · requirements.txt · packages.txt · runtime.txt · .env.example      (unchanged)
├── README.md · PROJECT_OVERVIEW.md · PROJECT_ANALYSIS.md · AGENTS.md                  (unchanged)
└── .gitignore · .dockerignore · .editorconfig · .gitattributes  (unchanged)
```

## Summary

| Kind | Count |
| --- | --- |
| Files moved (`git mv`) | 1 (`docs/migration_summary.md` → `docs/migration/migration_summary.md`) |
| Docs added | 5 |
| Docs updated | 1 (`folder_structure.md` — tree + v6 change log) |
| Code / imports / entry points / CI / Docker changed | 0 |
| Deleted | 0 |
