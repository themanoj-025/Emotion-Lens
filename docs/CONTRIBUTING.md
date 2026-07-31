# Contributing to Emotion-Lens

Thank you for your interest in contributing to Emotion-Lens, the real-time facial emotion recognition system!

## Getting Started

### Prerequisites
- Python 3.x
- pip

### Setup
1. Fork and clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (or use a pre-trained `emotion_model.h5`):
   ```bash
   python train.py --epochs 50
   ```

### Running the Applications

**Streamlit web app:**
```bash
streamlit run streamlit_app.py
```

**FastAPI server:**
```bash
uvicorn api_server:app --reload
```
OpenAPI docs available at `http://localhost:8000/docs`.

**Webcam inference (CLI):**
```bash
python webcam_inference.py
```

### Environment Variables (API server only)
| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI server port |

## Code Style

- Follow PEP 8 for Python code.
- Use descriptive variable names.
- Add docstrings to functions and classes.
- Keep TensorFlow/Keras model definitions clear and well-commented.

## Project Structure

- **`streamlit_app.py`** — Main Streamlit application entry point
- **`api_server.py`** — FastAPI REST API with Pydantic schemas
- **`inference.py`** — Core inference engine (load model, preprocess, predict)
- **`train.py`** — CNN training pipeline using FER2013 dataset
- **`pages/`** — Streamlit multipage UI modules (live camera, image analysis, analytics, etc.)
- **`utils/`** — Helper modules

### Model Details
- Input: 48×48 grayscale images
- Architecture: 3 Conv2D blocks (32→64→128 filters) → Flatten → Dense(1024) → Dropout(0.5) → Softmax(7)
- Emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Training dataset: FER2013 (downloaded via kagglehub)

## Running Tests

There is no formal test suite yet. Validation is done through:
- `TEST.ipynb` — Jupyter notebook for manual experimentation
- Running the Streamlit app and testing the UI manually
- Running the API and hitting `/health` and `/predict` endpoints

If you add new features, please include corresponding tests or update the notebook.

## Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make focused, minimal changes.
3. Verify the model still loads and predicts correctly.
4. If training-related changes are made, verify training completes without errors.
5. Commit with a descriptive message:
   - Format: `type(scope): description`
   - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
   - Example: `feat(api): add batch prediction endpoint`
   - Example: `fix(model): correct image preprocessing normalization`
6. Push and open a Pull Request.

## Reporting Issues

When reporting bugs, include:
- Steps to reproduce
- Error message and stack trace
- Whether the issue is in Streamlit, API, or training
- Model file status (does `emotion_model.h5` exist?)

## Training Notes

- The training script downloads the FER2013 dataset from Kaggle via `kagglehub`.
- Default: 50 epochs, batch size 64.
- Training requires approximately 2-4GB of RAM on CPU.
- The model is saved as `emotion_model.h5` in the project root.
- For better accuracy, consider adding data augmentation or more training epochs.

## API Extensions

When adding new API endpoints to `api_server.py`:
- Use Pydantic models for request/response validation.
- Follow the existing pattern (lazy model loading, structured error responses).
- Update the root info endpoint's endpoint list.
- Add CORS support if needed (already configured for all origins).

## Code of Conduct

This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.
