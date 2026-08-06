# API — EmotionLens: API Reference

| Field | Value |
| --- | --- |
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Backend Engineer |
| Status | In Review |

---

## 1. Endpoint Inventory

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/health` | None | Health check |
| POST | `/predict` | None | Predict from base64 image |
| POST | `/predict-file` | None | Predict from uploaded file |

## 2. Example: POST /predict

Request:

```json
{
  "image": "<base64-encoded image>"
}
```

Response:

```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "probabilities": {
    "angry": 0.01, "disgust": 0.02, "fear": 0.03,
    "happy": 0.87, "neutral": 0.03, "sad": 0.02, "surprise": 0.02
  }
}
```

## 3. Example: POST /predict-file

- Multipart form with `file=@face.jpg`.
- Response: same shape as /predict.

## 4. Error Codes

| Code | Meaning | Retry? |
| --- | --- | --- |
| 400 | Bad base64/file | No |
| 422 | Validation error | No |
| 500 | Model error | Yes |
| 503 | Model not loaded | Yes |

## 5. Rate Limits

None in v1 (single-user tool).

## 6. Auth Flow

N/A — no auth in v1 (public local/cloud inference).

## 7. Versioning Policy

- v1 flat paths; version prefix planned.

## 8. Related Documents

| Document | Relationship |
| --- | --- |
| [TechSpec.md](TechSpec.md) | API layer |
| [Schema.md](Schema.md) | Payloads |
| [SecurityAndCompliance.md](SecurityAndCompliance.md) | Upload policy |
| [AppFlow.md](../design/AppFlow.md) | Flows |
| [PRD.md](../product/PRD.md) | US-007 |
| [Design.md](../design/Design.md) | Response rendering |
| [ImplementationPlan.md](../project/ImplementationPlan.md) | TASK-3.3 |
| [Tracker.md](../project/Tracker.md) | Status |
| [Rules.md](../project/Rules.md) | Standards |
| [Testing.md](Testing.md) | Contract tests |
| [Deployment.md](Deployment.md) | Deploy |
| [Glossary.md](../reference/Glossary.md) | Vocabulary |
| [RiskRegister.md](../project/RiskRegister.md) | Risks |
