# Contributing to EmotionLens

Thanks for your interest in EmotionLens! Bug reports, documentation, and pull requests are welcome.

## Getting started

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feature/amazing`.
3. Install dependencies: `pip install -r requirements.txt`.

## Development workflow

- Add or update tests for every change and run them with `pytest`.
- Verify the app boots: `streamlit run streamlit_app.py` (dashboard) or `python api_server.py` (API).
- Keep code consistent with the existing style in `utils/` and the `pages/` modules.

## Commit conventions

Keep commits small and focused. Prefix messages with a type, e.g. `feat:`, `fix:`, `docs:`, `test:`.

## Opening a pull request

1. Push your branch and open a PR against `main`.
2. Describe what you changed and why.
3. Link any related issue.

By contributing, you agree that your contributions are licensed under the MIT License.
