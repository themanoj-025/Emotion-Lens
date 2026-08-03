# Emotion-Lens — Docker Guide

## Quick start

```bash
docker compose up -d
```

One service (`app`) serves the Streamlit UI on **http://localhost:8501**.
**First prediction downloads the model checkpoint via `kagglehub`** — the
container needs outbound network access the first time it runs.

## Environment

No secrets are required. The model is fetched at runtime; `kagglehub`
uses its own cache (inside the container).

## Development

```bash
make up   # dev override: bind mounts + hot reload (polling watcher)
```

## Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Restart `always`, 2G memory limit, no dev mounts.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Model not found" on first prediction | Ensure egress network access; kagglehub downloads on first use |
| Blank page | Give the healthcheck `start_period` time (60s) while TF loads |
| Port 8501 in use | Change `ports` in `docker-compose.yml` |
