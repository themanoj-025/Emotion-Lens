"""API routes -- prediction, health, and info endpoints."""



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
async def metrics():
    """Prometheus metrics endpoint."""
    if not _PROM_AVAILABLE:
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
