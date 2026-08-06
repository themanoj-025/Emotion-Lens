# SecurityAndCompliance — EmotionLens: Security & Privacy

| Field | Value |
|---|---|
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Security Engineer |
| Status | In Review |

---

## 1. Threat Model (STRIDE)

| Threat | Surface | Impact | Mitigation |
|---|---|---|---|
| Spoofing | API abuse | Resource use | (v1 public; optional key) |
| Tampering | Uploaded images | Bad predictions | Pydantic validation + size caps |
| Info disclosure | Camera frames | Privacy | Frames never persisted |
| DoS | Predict flood | CPU exhaustion | Size caps; single-user scope |
| Elevation | — | — | N/A (no roles) |

## 2. Auth / Authorization

- No auth in v1 (local/single-user tool).
- Optional API key flag (roadmap).

## 3. Data Classification

| Data | Class | Handling |
|---|---|---|
| Webcam frames | biometric-ish | in-memory only, never persisted |
| Uploaded images | personal | session-scoped, not stored |
| Predictions | none | analytics optional |
| Model file | internal | bundled artifact |

## 4. Encryption

- In transit: TLS on hosted deployment.
- At rest: nothing sensitive persisted.

## 5. Compliance Checklist

- [ ] Camera frames not persisted
- [ ] Upload size caps
- [ ] No PII collection
- [ ] Model license documented (FER2013 academic use)

## 6. Incident Response Plan (outline)

1. Detect: hosting abuse alerts.
2. Triage: abuse vs bug.
3. Contain: disable endpoint.
4. Remediate + tests.
5. Recover.
6. Postmortem.

## 7. Related Documents

| Document | Relationship |
|---|---|
| [Rules.md](Rules.md) | Privacy rules |
| [API.md](API.md) | Endpoints |
| [Schema.md](Schema.md) | Sensitive map |
| [TechSpec.md](TechSpec.md) | NFRs |
| [PRD.md](PRD.md) | Goals |
| [AppFlow.md](AppFlow.md) | Flows |
| [Design.md](Design.md) | Design |
| [ImplementationPlan.md](ImplementationPlan.md) | Tasks |
| [Tracker.md](Tracker.md) | Status |
| [Testing.md](Testing.md) | Security tests |
| [Deployment.md](Deployment.md) | Deploy |
| [Glossary.md](Glossary.md) | Vocabulary |
| [RiskRegister.md](RiskRegister.md) | Risks |
