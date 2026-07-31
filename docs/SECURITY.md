# Security Policy for Emotion-Lens

## Reporting a Vulnerability

If you discover a security vulnerability in Emotion-Lens, please report it privately.

**How to report:**
- Open a private security advisory on GitHub (if this repository is public).
- Email **manojjana.0025@gmail.com** directly. This contact is also listed in our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- If neither channel works, open a standard issue with the label `security` without including exploit details.

**Expectations:**
- We will acknowledge receipt within 5 business days.
- We will assess and respond within 10 business days.

## Security Posture

**⚠️ Important:** Emotion-Lens is a local/development-grade application. It is **not designed for production deployment** without significant security hardening. The following sections outline current security measures and critical gaps.

## Security Measures

### Implemented
- **None.** This project has no authentication, no authorization, no rate limiting, and no input sanitization beyond basic image format validation via OpenCV.

### Not Implemented (Critical Gaps)
- **No authentication:** Both the Streamlit app and FastAPI API are open to all connections. Anyone who can reach the server can use all features.
- **CORS wide open:** `allow_origins=["*"]` — any website can make cross-origin requests to the API.
- **No rate limiting:** The API can be called unlimited times, posing a DoS risk.
- **No HTTPS:** All communication is plain HTTP.
- **No input sanitization:** Beyond basic image validation, no content filtering is applied.
- **No user isolation:** All users share the same model and have access to the same functionality.

## Recommended Hardening for Production

If deploying Emotion-Lens in any production or public-facing environment, you **must** implement at minimum:

1. **Authentication:** Add API key or token-based auth to the FastAPI server.
2. **Rate limiting:** Implement request throttling (e.g., via slowapi or a reverse proxy).
3. **CORS restriction:** Restrict `allow_origins` to specific trusted domains.
4. **HTTPS:** Deploy behind a reverse proxy with TLS termination (nginx, Caddy).
5. **Input validation:** Add image content-type verification and size limits.
6. **Container security:** If using Docker, run the container as a non-root user.

## API Security

The FastAPI server (`api_server.py`) currently has:
- Open CORS policy (`allow_origins=["*"]`)
- No authentication on any endpoint (`/predict`, `/predict-file`, `/health`)
- No request size limits
- No file type restrictions beyond content-type check

**Do not expose this API to the public internet without implementing the hardening steps above.**

## Model Security

- The trained model (`emotion_model.h5`) is a standard Keras H5 file.
- Model files do not contain executable code, but treat them as untrusted data if sourced externally.
- The model runs entirely on-device — no data is sent to external services.

## Environment Variables

| Variable | Sensitivity | Notes |
|---|---|---|
| `API_HOST` | Low | Bind address (default: `0.0.0.0`). Change to `127.0.0.1` for local-only access. |
| `API_PORT` | Low | Server port (default: `8000`). |

The `API_HOST` variable defaults to `0.0.0.0`, which listens on all network interfaces. For local-only use, set `API_HOST=127.0.0.1`.

## Dependency Security

This project uses TensorFlow/Keras, which is a large dependency with a significant attack surface. Keep dependencies updated:

```bash
pip-audit -r requirements.txt
```
