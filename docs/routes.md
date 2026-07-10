# Routes — Emotion-Lens

## Streamlit Pages (Frontend)
| Page | Route (via sidebar) | Description |
|------|---------------------|-------------|
| 🎥 Live Camera | "🎥 Live Camera" | Real-time webcam detection |
| 🖼️ Image Analysis | "🖼️ Image Analysis" | Upload images for analysis |
| 📊 Analytics | "📊 Analytics" | Session analytics dashboard |
| 🏋️ Train Model | "🏋️ Train Model" | GUI-based model training |
| 🧠 Model Inspector | "🧠 Model Inspector" | View model architecture |
| 🎯 Emotion Game | "🎯 Emotion Game" | Interactive emotion challenge |
| 📖 About | "📖 About" | Project information |

## FastAPI API Routes
| Method | Route | Description | Auth |
|--------|-------|-------------|------|
| GET | `/` | API root with endpoint listing | No |
| GET | `/health` | Health check with model status | No |
| POST | `/predict` | Predict from base64 image | No |
| POST | `/predict-file` | Predict from uploaded file | No |
| GET | `/docs` | Swagger UI docs | No |
| GET | `/redoc` | ReDoc documentation | No |
