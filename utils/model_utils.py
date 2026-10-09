"""
Model utility functions for EmotionLens 🎭
Handles model loading with Streamlit caching, face cascade loading,
and automatic model download for Streamlit Cloud deployment.
"""

import contextlib
import os
import time

import cv2
import streamlit as st
from tensorflow.keras.models import load_model

# Emotion mapping - must match FER2013 training order
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Emotion → Color → Emoji mapping
EMOTION_CONFIG = {
    "Angry": {"color": "#FF6B6B", "emoji": "😠", "bg": "#2D1515"},
    "Disgust": {"color": "#9B59B6", "emoji": "🤢", "bg": "#1E0A2E"},
    "Fear": {"color": "#F39C12", "emoji": "😨", "bg": "#2D1F0A"},
    "Happy": {"color": "#2ECC71", "emoji": "😊", "bg": "#0A2D15"},
    "Neutral": {"color": "#95A5A6", "emoji": "😐", "bg": "#1A1F20"},
    "Sad": {"color": "#3498DB", "emoji": "😢", "bg": "#0A1520"},
    "Surprise": {"color": "#E67E22", "emoji": "😲", "bg": "#2D1A0A"},
}

MODEL_PATH = "emotion_model.h5"


@st.cache_resource
def load_model_cached(path=None):
    """
    Load the Keras emotion detection model once, cached for all pages.

    Args:
        path: Path to the .h5 model file. Defaults to 'emotion_model.h5'

    Returns:
        Loaded Keras model or None if not found
    """
    if path is None:
        path = MODEL_PATH

    if not os.path.exists(path):
        st.error(f"❌ Model file not found at: {os.path.abspath(path)}")
        st.info(
            "Please train a model first using the Train Model page or place a trained `.h5` file in the project root."
        )
        return None

    try:
        model = load_model(path)
        return model
    except (OSError, ValueError) as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_resource
def load_face_cascade():
    """
    Load OpenCV Haar Cascade for face detection (cached).

    Returns:
        cv2.CascadeClassifier or None if failed
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("❌ Failed to load face cascade classifier.")
            return None
        return face_cascade
    except (OSError, ValueError) as e:
        st.error(f"❌ Error loading face cascade: {e}")
        return None


def is_model_available():
    """Check if the model file exists on disk."""
    return os.path.exists(MODEL_PATH)


# Model Auto-Download (for Streamlit Cloud)

# Add URLs to pre-trained model files here to enable auto-download on Streamlit Cloud.
# Example:
# MODEL_DOWNLOAD_URLS = [
#     "https://github.com/YOUR_USER/YOUR_REPO/releases/download/v1.0/emotion_model.h5",
# ]
MODEL_DOWNLOAD_URLS: list[str] = []

# Download hardening — bounded retries so a flaky network during a Streamlit
# Cloud cold start doesn't hard-fail the app. Retries only cover transient
# failures (network exceptions, 429/5xx); a 404 or a disk error fails fast.
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_SECONDS = (2, 5, 10)
_DOWNLOAD_CONNECT_TIMEOUT = 10  # seconds — fail fast on unreachable hosts
_DOWNLOAD_READ_TIMEOUT = 120  # seconds between chunks for large files
_TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
_MIN_MODEL_BYTES = 100_000


def try_download_model() -> bool:
    """Attempt to download a pre-trained emotion model from configured URLs.

    Tries each URL in MODEL_DOWNLOAD_URLS with bounded retries and
    exponential backoff, so a flaky network at Streamlit Cloud boot doesn't
    take the app down permanently. Bytes stream to a temp file that is only
    moved into place after validation (atomic publish), so a failed download
    can never leave a corrupt ``emotion_model.h5`` behind.

    Returns True if a model was successfully downloaded and saved, False
    otherwise. This is automatically called on app startup in Streamlit Cloud.
    """
    if is_model_available():
        return True

    if not MODEL_DOWNLOAD_URLS:
        return False

    try:
        import requests
    except ImportError:
        return False

    timeout: tuple[float, float] = (_DOWNLOAD_CONNECT_TIMEOUT, _DOWNLOAD_READ_TIMEOUT)

    for url in MODEL_DOWNLOAD_URLS:
        tmp_path = MODEL_PATH + ".part"
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            try:
                st.info(
                    f"⬇️ Downloading model from {url} "
                    f"(attempt {attempt + 1}/{_DOWNLOAD_ATTEMPTS})..."
                )
                with requests.get(url, timeout=timeout, stream=True) as resp:
                    if resp.status_code in _TRANSIENT_HTTP_STATUS:
                        st.warning(f"⚠️ HTTP {resp.status_code} from {url}; will retry.")
                        time.sleep(
                            _DOWNLOAD_BACKOFF_SECONDS[
                                min(attempt, len(_DOWNLOAD_BACKOFF_SECONDS) - 1)
                            ]
                        )
                        continue
                    if resp.status_code != 200:
                        st.warning(f"⚠️ HTTP {resp.status_code} from {url}; not retrying.")
                        break
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                if os.path.getsize(tmp_path) <= _MIN_MODEL_BYTES:
                    st.warning("⚠️ Downloaded file failed validation (too small); will retry.")
                else:
                    os.replace(tmp_path, MODEL_PATH)  # atomic publish
                    st.success(
                        f"✅ Model downloaded successfully ({os.path.getsize(MODEL_PATH) // 1024} KB)"
                    )
                    return True
            except requests.RequestException as e:
                st.warning(f"⚠️ Download attempt {attempt + 1} failed: {e}")
            except OSError as e:
                # Disk/write errors won't be fixed by retrying this URL.
                st.warning(f"⚠️ Could not write model file: {e}")
                break

            if attempt < _DOWNLOAD_ATTEMPTS - 1:
                time.sleep(
                    _DOWNLOAD_BACKOFF_SECONDS[min(attempt, len(_DOWNLOAD_BACKOFF_SECONDS) - 1)]
                )

        # Never leave a partial download behind — a truncated emotion_model.h5
        # would make is_model_available() lie on the next boot.
        if os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)

    return False


def ensure_model_on_cloud():
    """Run on Streamlit Cloud startup to ensure model is available.

    Checks if the model exists. If not, tries to download or prompts
    the user to train one.
    """
    if is_model_available():
        return

    # Attempt auto-download
    if try_download_model():
        return

    # Last resort: prompt the user
    st.warning(
        "⚠️ **No pre-trained model found.**\n\n"
        "To use EmotionLens on Streamlit Cloud, you have two options:\n\n"
        "1. **Train a model here** — Go to the **🏋️ Train Model** page and train one. "
        "The FER2013 dataset will be auto-downloaded. Training takes ~10–30 minutes.\n\n"
        "2. **Upload your own model** — Add your `emotion_model.h5` file to the project root "
        "and redeploy.\n\n"
        "You can also deploy the model separately by setting up a GitHub Release "
        "and adding the URL to `MODEL_DOWNLOAD_URLS` in `model_utils.py`."
    )


def get_model_summary(model):
    """
    Get model summary as a list of layer dictionaries.

    Args:
        model: Loaded Keras model

    Returns:
        List of dicts with layer info, and param counts
    """
    layers_info = []
    for layer in model.layers:
        layer_dict = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(layer.output_shape) if hasattr(layer, "output_shape") else "—",
            "params": layer.count_params(),
        }
        layers_info.append(layer_dict)

    total_params = model.count_params()
    trainable_params = sum(
        layer.count_params()
        for layer in model.layers
        if hasattr(layer, "trainable") and layer.trainable
    )
    non_trainable_params = total_params - trainable_params

    return layers_info, {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


# Mood Music Sync — Spotify/YouTube search queries per emotion
MOOD_MUSIC_MAP = {
    "Angry": {
        "spotify": "angry heavy metal rage playlist",
        "youtube": "angry heavy metal rage songs",
        "spotify_uri": "spotify:search:angry+heavy+metal+rage",
        "vibe": "🤘 Rage & Energy",
        "desc": "Channel that anger into heavy riffs and pounding drums",
    },
    "Disgust": {
        "spotify": "dark ambient disturbing playlist",
        "youtube": "dark disturbing ambient music",
        "spotify_uri": "spotify:search:dark+ambient+disturbing",
        "vibe": "🖤 Dark & Disturbed",
        "desc": "Embrace the darkness with eerie ambient soundscapes",
    },
    "Fear": {
        "spotify": "creepy horror suspense playlist",
        "youtube": "creepy horror suspense music",
        "spotify_uri": "spotify:search:creepy+horror+suspense",
        "vibe": "👻 Suspense & Horror",
        "desc": "Let the tension build with spine-chilling soundtracks",
    },
    "Happy": {
        "spotify": "happy upbeat feel good playlist",
        "youtube": "happy upbeat feel good songs",
        "spotify_uri": "spotify:search:happy+upbeat+feel+good",
        "vibe": "🌞 Feel-Good Vibes",
        "desc": "Ride that happiness with uplifting beats and sunny melodies",
    },
    "Neutral": {
        "spotify": "chill ambient study focus",
        "youtube": "chill lo-fi music study",
        "spotify_uri": "spotify:search:chill+ambient+study",
        "vibe": "🧘 Chill & Focused",
        "desc": "Stay centered with calm lo-fi beats and ambient textures",
    },
    "Sad": {
        "spotify": "sad melancholic cry playlist",
        "youtube": "sad melancholic songs playlist",
        "spotify_uri": "spotify:search:sad+melancholic+cry",
        "vibe": "💧 Melancholy & Reflection",
        "desc": "Let it out with soul-stirring ballads and melancholic melodies",
    },
    "Surprise": {
        "spotify": "epic cinematic orchestral playlist",
        "youtube": "epic cinematic orchestral music",
        "spotify_uri": "spotify:search:epic+cinematic+orchestral",
        "vibe": "🎬 Epic & Cinematic",
        "desc": "Feel the awe with soaring orchestral epics and dramatic builds",
    },
}


# Plotly color theme for consistent chart styling
PLOTLY_THEME = {
    "paper_bgcolor": "#1C2128",
    "plot_bgcolor": "#161B22",
    "font": {"color": "#E6EDF3", "family": "Inter, sans-serif"},
}
