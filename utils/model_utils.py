"""
Model utility functions for EmotionLens 🎭
Handles model loading with Streamlit caching, face cascade loading.
"""

import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Emotion mapping - must match FER2013 training order
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion → Color → Emoji mapping
EMOTION_CONFIG = {
    'Angry':    {'color': '#FF6B6B', 'emoji': '😠', 'bg': '#2D1515'},
    'Disgust':  {'color': '#9B59B6', 'emoji': '🤢', 'bg': '#1E0A2E'},
    'Fear':     {'color': '#F39C12', 'emoji': '😨', 'bg': '#2D1F0A'},
    'Happy':    {'color': '#2ECC71', 'emoji': '😊', 'bg': '#0A2D15'},
    'Neutral':  {'color': '#95A5A6', 'emoji': '😐', 'bg': '#1A1F20'},
    'Sad':      {'color': '#3498DB', 'emoji': '😢', 'bg': '#0A1520'},
    'Surprise': {'color': '#E67E22', 'emoji': '😲', 'bg': '#2D1A0A'},
}

MODEL_PATH = 'emotion_model.h5'


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
        st.info("Please train a model first using the Train Model page or place a trained `.h5` file in the project root.")
        return None
    
    try:
        model = load_model(path)
        return model
    except Exception as e:
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
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("❌ Failed to load face cascade classifier.")
            return None
        return face_cascade
    except Exception as e:
        st.error(f"❌ Error loading face cascade: {e}")
        return None


def is_model_available():
    """Check if the model file exists on disk."""
    return os.path.exists(MODEL_PATH)


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
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else '—',
            'params': layer.count_params(),
        }
        layers_info.append(layer_dict)
    
    total_params = model.count_params()
    trainable_params = sum(
        layer.count_params() for layer in model.layers 
        if hasattr(layer, 'trainable') and layer.trainable
    )
    non_trainable_params = total_params - trainable_params
    
    return layers_info, {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
    }


# Mood Music Sync — Spotify/YouTube search queries per emotion
MOOD_MUSIC_MAP = {
    'Angry': {
        'spotify': 'angry heavy metal rage playlist',
        'youtube': 'angry heavy metal rage songs',
        'spotify_uri': 'spotify:search:angry+heavy+metal+rage',
        'vibe': '🤘 Rage & Energy',
        'desc': 'Channel that anger into heavy riffs and pounding drums',
    },
    'Disgust': {
        'spotify': 'dark ambient disturbing playlist',
        'youtube': 'dark disturbing ambient music',
        'spotify_uri': 'spotify:search:dark+ambient+disturbing',
        'vibe': '🖤 Dark & Disturbed',
        'desc': 'Embrace the darkness with eerie ambient soundscapes',
    },
    'Fear': {
        'spotify': 'creepy horror suspense playlist',
        'youtube': 'creepy horror suspense music',
        'spotify_uri': 'spotify:search:creepy+horror+suspense',
        'vibe': '👻 Suspense & Horror',
        'desc': 'Let the tension build with spine-chilling soundtracks',
    },
    'Happy': {
        'spotify': 'happy upbeat feel good playlist',
        'youtube': 'happy upbeat feel good songs',
        'spotify_uri': 'spotify:search:happy+upbeat+feel+good',
        'vibe': '🌞 Feel-Good Vibes',
        'desc': 'Ride that happiness with uplifting beats and sunny melodies',
    },
    'Neutral': {
        'spotify': 'chill ambient study focus',
        'youtube': 'chill lo-fi study music',
        'spotify_uri': 'spotify:search:chill+ambient+study',
        'vibe': '🧘 Chill & Focused',
        'desc': 'Stay centered with calm lo-fi beats and ambient textures',
    },
    'Sad': {
        'spotify': 'sad melancholic cry playlist',
        'youtube': 'sad melancholic songs playlist',
        'spotify_uri': 'spotify:search:sad+melancholic+cry',
        'vibe': '💧 Melancholy & Reflection',
        'desc': 'Let it out with soul-stirring ballads and melancholic melodies',
    },
    'Surprise': {
        'spotify': 'epic cinematic orchestral playlist',
        'youtube': 'epic cinematic orchestral music',
        'spotify_uri': 'spotify:search:epic+cinematic+orchestral',
        'vibe': '🎬 Epic & Cinematic',
        'desc': 'Feel the awe with soaring orchestral epics and dramatic builds',
    },
}


# Plotly color theme for consistent chart styling
PLOTLY_THEME = {
    'paper_bgcolor': '#1C2128',
    'plot_bgcolor': '#161B22',
    'font': {'color': '#E6EDF3', 'family': 'Inter, sans-serif'},
}
