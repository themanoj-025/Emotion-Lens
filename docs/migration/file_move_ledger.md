# File Move Ledger — Emotion-Lens

Restructure date: **2026-08-11** (v6) · Method: `git mv` · Branch: `main`
(local commits, no push).

## Moved Files

| # | Old Path | New Path | Category | Reason | Risk | Verified? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `docs/migration_summary.md` | `docs/migration/migration_summary.md` | Meta → Docs | Consolidate migration records under `docs/migration/` (protocol Phase 6) | Low (0 refs; grep-verified) | ✅ |

## Files Added

| Path | Reason |
| --- | --- |
| `docs/module_dependency.md` | Phase 6 deliverable. |
| `docs/startup_flow.md` | Phase 6 deliverable. |
| `docs/package_overview.md` | Phase 6 deliverable. |
| `docs/migration/old_tree_to_new_tree.md` | Phase 6 deliverable. |
| `docs/migration/file_move_ledger.md` | Phase 6 deliverable (this file). |

## Files Updated

| Path | Reason |
| --- | --- |
| `docs/folder_structure.md` | Docs tree now lists `migration/`; added v6 change-log section. |

## Files Deliberately NOT MOVED (contract analysis)

| Path | Why it stays | Risk if moved |
| --- | --- | --- |
| `streamlit_app.py` | Streamlit entry — Docker CMD, Cloud entry config, CI pyflakes matrix | High |
| `pages/` | Must be a sibling of the entry script (Streamlit multipage discovery) | High — breaks navigation |
| `utils/` | Importable top-level package — CI has per-module `from utils.* import ...` checks; all pages import it | High — ~20 imports + ~40 CI refs |
| `api_server.py`, `train.py`, `inference.py`, `webcam_inference.py` | Standalone entry points — Docker COPY, CI pyflakes list | Medium |
| `assets/`, `.streamlit/`, `.devcontainer/` | Framework lookup paths (style, config, devcontainer) | Medium |
| `saved_model/` | Runtime artifacts — gitignored by design | — |

The v5.0 pass reached the same conclusion ("no files moved — structure already
consistent with target architecture"); this restructure re-affirms it with the
framework-contract evidence above.

## Flagged (follow-up backlog)

| Item | Flag |
| --- | --- |
| **No pytest suite** | CI is syntax/import/security/build gated only. Recommended unit tests for pure `utils/` modules (`emotion_utils`, `smoothing_utils`, `session_utils`, `config`) — all stdlib/mockable without TensorFlow. |
| `api_server.py` upload-size limits | Pre-existing flag from v5.0 summary (upload size cap behavior). |
| TensorFlow import weight in `utils/model_utils.py`, `utils/gradcam_utils.py` | Kept separate from pure modules so CI no-model checks stay fast — document this boundary in CODEOWNERS/PR guidance if the suite grows. |

## Deletions

None in this restructure.
