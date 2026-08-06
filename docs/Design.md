# Design — EmotionLens: Design System & UX Principles

| Field | Value |
|---|---|
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Design Lead |
| Status | In Review |

---

## 1. Design Principles

1. **Live-first** — camera/detection is the hero experience.
2. **Visual feedback** — charts and overlays explain instantly.
3. **Playful but serious** — game features engage without gimmick.
4. **Consistent** — shared components across 8 pages.
5. **Clear errors** — setup/limitations are honest.

## 2. Brand & Visual Identity

- Voice: friendly, educational, CV-focused.
- Imagery: live camera feed, Grad-CAM heatmaps, radar charts.

## 3. Color System

| Token | Hex | Usage | Contrast (AA) |
|---|---|---|---|
| bg | `#0E1117` | Streamlit dark bg | — |
| surface | `#262730` | cards | — |
| text | `#FAFAFA` | body | 15:1 |
| accent | `#FF4B4B` | Streamlit red CTAs | 5.2:1 |
| emotion-happy | `#22C55E` | Happy badge | 5:1 |
| emotion-sad | `#3B82F6` | Sad badge | 5.8:1 |
| emotion-anger | `#EF4444` | Angry badge | 5.5:1 |
| emotion-neutral | `#94A3B8` | Neutral badge | 7:1 |

## 4. Typography Scale

| Token | Font | Size | Weight | Line-height | Usage |
|---|---|---|---|---|---|
| display | system sans | 28px | 700 | 1.2 | page titles |
| heading | system sans | 20px | 600 | 1.3 | sections |
| body | system sans | 14px | 400 | 1.5 | content |
| emotion-label | system sans | 16px | 700 | 1.3 | overlay labels |
| caption | system sans | 12px | 400 | 1.4 | meta |

## 5. Spacing & Grid

- Base 4px; Streamlit layout.
- Breakpoints: Streamlit responsive.

## 6. Component Library

**Emotion overlay (live camera):**

```
┌────────────────────────────┐
│ [camera feed]              │
│ ┌────────┐                 │
│ │ [face] │  😊 Happy 0.87 │
│ │ box    │  ──────────────│
│ └────────┘  prob bar      │
└────────────────────────────┘
```

**Radar chart** (image analysis): Plotly radar of 7-class probabilities.

Other: sidebar nav, KPI card, training config panel, progress bar, game scoreboard.

## 7. Iconography

Emoji + Plotly; no image assets.

## 8. Accessibility

- WCAG 2.1 AA targets.
- Emotion never by color alone (labels included).
- Keyboard nav.

## 9. Responsive

- Streamlit fluid; camera feeds scale.

## 10. Motion

- Chart transitions (300ms); detection updates animated lightly; reduced-motion honored.

## 11. Dark Mode

Dark theme default (Streamlit dark).

## 12. Related Documents

| Document | Relationship |
|---|---|
| [AppFlow.md](AppFlow.md) | Screens |
| [PRD.md](PRD.md) | UX goals |
| [TechSpec.md](TechSpec.md) | Stack |
| [Schema.md](Schema.md) | Session data |
| [ImplementationPlan.md](ImplementationPlan.md) | Tasks |
| [Tracker.md](Tracker.md) | Status |
| [Rules.md](Rules.md) | Standards |
| [API.md](API.md) | Contracts |
| [SecurityAndCompliance.md](SecurityAndCompliance.md) | Privacy |
| [Testing.md](Testing.md) | UI tests |
| [Deployment.md](Deployment.md) | Deploy |
| [Glossary.md](Glossary.md) | Vocabulary |
| [RiskRegister.md](RiskRegister.md) | Risks |
