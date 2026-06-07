"""
Session state management utilities for EmotionLens 🎭
Handles initialization, prediction recording, and data export.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
from io import StringIO, BytesIO


def init_session_state():
    """
    Initialize all session state variables.
    Safe to call on every page — only sets missing keys.
    """
    defaults = {
        'predictions': [],           # List of {emotion, confidence, probabilities, timestamp}
        'snapshots': [],             # Captured frames as PIL Images with metadata
        'session_start': datetime.now(),
        'total_predictions': 0,
        'game_score': 0,
        'game_high_score': 0,
        'game_history': [],          # List of game results
        'emotion_history': [],       # Rolling history for live smoothing
        'temporal_buffer': [],       # Buffer for temporal smoothing
        'locked_emotion': None,      # Locked frame comparison
        'model_path': 'emotion_model.h5',
        'camera_active': False,
        'dark_mode': True,           # Theme toggle state
        # Timeline Replay state
        'timeline_recording': False,
        'timeline_start_idx': 0,
        'timeline_recordings': [],   # List of {name, predictions_slice, start_time, duration}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_prediction(emotion, confidence, all_probs):
    """
    Record a prediction in session state.
    
    Args:
        emotion: Predicted emotion string
        confidence: Confidence score (0-1)
        all_probs: Array/list of all 7 probabilities
    """
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    
    st.session_state.predictions.append({
        'emotion': emotion,
        'confidence': float(confidence),
        'probabilities': all_probs.tolist() if hasattr(all_probs, 'tolist') else list(all_probs),
        'timestamp': datetime.now().isoformat(),
    })
    st.session_state.total_predictions += 1


def add_snapshot(pil_image, emotion, confidence):
    """Add a snapshot to session state with metadata."""
    if 'snapshots' not in st.session_state:
        st.session_state.snapshots = []
    
    st.session_state.snapshots.append({
        'image': pil_image,
        'emotion': emotion,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat(),
    })


def get_prediction_dataframe():
    """Get predictions as a pandas DataFrame for analysis/export."""
    if not st.session_state.get('predictions'):
        return pd.DataFrame()
    
    df = pd.DataFrame(st.session_state.predictions)
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_emotion_distribution():
    """Get emotion count distribution from session predictions."""
    df = get_prediction_dataframe()
    if df.empty:
        return {}
    return df['emotion'].value_counts().to_dict()


def export_predictions_csv():
    """Export predictions as downloadable CSV content."""
    df = get_prediction_dataframe()
    if df.empty:
        return None
    
    # Remove probabilities array for CSV cleanliness
    export_df = df.drop(columns=['probabilities'], errors='ignore')
    csv_buffer = StringIO()
    export_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def export_predictions_json():
    """Export predictions as downloadable JSON content."""
    if not st.session_state.get('predictions'):
        return None
    return json.dumps(st.session_state.predictions, indent=2, default=str)


def reset_session():
    """Reset all session data but keep preferences."""
    keys_to_keep = {'model_path', 'dark_mode', 'game_high_score'}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    init_session_state()


def format_session_duration():
    """Return formatted session duration string."""
    if 'session_start' not in st.session_state:
        return "00:00:00"
    
    delta = datetime.now() - st.session_state.session_start
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
