# ✦ AI Assistance Disclosure

## Summary

Emotion-Lens was built with substantial AI assistance. The FastAPI
application, transformer-based emotion classifier, preprocessing and
tokenization pipeline, and the Streamlit dashboard were drafted
end-to-end by AI. Human maintainers then reviewed every design decision,
validated the model metrics, and wired up the CI and security controls.

## Tools & Models Used

| Tool | Provider / Model | Interface | Approx. dates used |
|---|---|---|---|
| **Claude / Opus** | Anthropic | CLI agent + web chat | 2025-11 → 2026-09 |
| **Cursor** | Anthropic (via IDE) | IDE agent | 2026-03 → 2026-09 |
| **GitHub Copilot** | GitHub | IDE completion | 2025-11 → 2026-09 |

## Scope — What AI Did vs. What Humans Did

- **AI:**
  - First drafts of the FastAPI app, emotion model, and NLP preprocessing
    pipeline.
  - Initial drafts of `README.md`, `CONTRIBUTING.md`, and the dashboard.
- **Human:**
  - All architecture decisions (model choice, multi-label handling).
  - Security review — secrets handling, input validation, rate limiting.
  - All model evaluation, test suite, CI workflows, and production
    hardening.
  - The open-source stewardship.

## Estimated AI-Assisted Share

Roughly **60–75%** of lines in `api/`, `model/`, `nlp/`, and
`dashboard/` follow AI-generated boilerplate. The business logic,
security controls, and tests are mostly human-authored. This is an
estimate, not a measured statistic — the exact ratio was never
instrumented.

## Human Review Process

- Every PR is reviewed line-by-line by the maintainer.
- Security review by maintainer: secrets handling, input validation,
  gitleaks/bandit findings.
- Automated gates: pre-commit (ruff + mypy), pytest (≥ 80% coverage
  floor), gitleaks, trivy, and an AI-disclosure metadata check.

## Known Limitations & Risks

- **LLM hallucination:** the model may mislabel an emotion; the
  per-label threshold calibration is manually tuned.
- **Dataset bias:** the emotion dataset skews toward certain labels;
  evaluation metrics and a confusion matrix catch drift.
- **License-contamination risk:** upstream training data may contain code
  copies. The CI gitleaks/bandit gates catch obvious secrets.

## How to Verify

- `git log --all --oneline --grep="assistant\\|ai\\|gen"` and the
  `Co-authored-by:` trailers in recent commits.
- `cat docs/audit/*` — per-repo audit reports.
- `git log --all --oneline --grep="ai-assisted\\|LLM\\|llm"` — the commit
  history trail.

## Last Updated

2026-10-06 · Maintained by `themanoj-025 <code.me.025@gmail.com>`
