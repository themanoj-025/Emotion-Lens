"""API routes -- root, health, and metrics endpoints.

These unversioned routes are registered on the ``app`` created in
``api_server``. The ``from api_server import app`` below is circular-safe:
api_server defines ``app`` before it reaches ``from api_routes import *``.
"""

import os

from fastapi.responses import Response

from api_models import EMOTIONS, MODEL_PATH, HealthResponse
from api_server import _PROM_AVAILABLE, app
from inference import get_model

try:
    from prometheus_client import generate_latest
except ImportError:
    generate_latest = None  # prometheus_client is optional


# Root / health / metrics (unversioned — for probes and monitoring)


@app.get("/", tags=["Info"])
async def root() -> None:
    """API root — provides basic info and links."""
    return {
        "service": "EmotionLens 🎭 API",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict": "Predict emotion from base64 image",
            "POST /predict-file": "Predict emotion from uploaded file",
            "GET  /docs": "Swagger UI documentation",
            "GET  /redoc": "ReDoc documentation",
        },
        "emotions": EMOTIONS,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> None:
    """Health check endpoint. Confirms the server and model are operational."""
    model, _cascade = get_model()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=os.path.abspath(MODEL_PATH) if os.path.exists(MODEL_PATH) else "NOT FOUND",
        emotions=EMOTIONS,
    )


@app.get("/metrics", tags=["Info"])
async def metrics() -> dict:
    """Prometheus metrics endpoint."""
    if not _PROM_AVAILABLE or generate_latest is None:
        return {"status": "prometheus_client not installed"}
    return Response(content=generate_latest(), media_type="text/plain")


# CLI Entry Point

if __name__ == "__main__":
    import uvicorn

    print(
        f"""
╔══════════════════════════════════════════════════════════╗
║              EmotionLens 🎭  API Server                  ║
╠══════════════════════════════════════════════════════════╣
║  Endpoints:                                              ║
║    • Health:   http://{HOST}:{PORT}/health                    ║
║    • Predict:  POST http://{HOST}:{PORT}/predict              ║
║    • Upload:   POST http://{HOST}:{PORT}/predict-file         ║
║    • Docs:     http://{HOST}:{PORT}/docs                      ║
╠══════════════════════════════════════════════════════════╣
║  Model: {MODEL_PATH:<46}║
║  Emotions: {", ".join(EMOTIONS)}  ║
╚══════════════════════════════════════════════════════════╝
    """
    )

    uvicorn.run(app, host=HOST, port=PORT)
