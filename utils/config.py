"""
EmotionLens 🎭 — Global Configuration
All constants, emotion mappings, and shared configuration values.
Import from here instead of hardcoding values across pages.
"""
from __future__ import annotations
from typing import Final

# ─── Emotion Labels (MUST match FER2013 training order) ─────
# Softmax index 0 = Angry, index 6 = Surprise
EMOTIONS: Final = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ─── Model Paths ────────────────────────────────────────────
MODEL_PATH: Final = 'models/emotion_model.h5'
IMG_SIZE: Final = (48, 48)

# ─── Emotion Config (color, bg, emoji, valence, arousal) ───
# valence: -1.0 (negative) to +1.0 (positive)
# arousal: 0.0 (calm) to 1.0 (excited)
EMOTION_CONFIG: Final = {
    'Angry':    {'color': '#FF6B6B', 'bg': '#1E0808', 'emoji': '😠', 'valence': -1.0, 'arousal':  0.9},
    'Disgust':  {'color': '#A855F7', 'bg': '#150A26', 'emoji': '🤢', 'valence': -0.8, 'arousal':  0.4},
    'Fear':     {'color': '#FB923C', 'bg': '#1E1008', 'emoji': '😨', 'valence': -0.7, 'arousal':  0.8},
    'Happy':    {'color': '#4ADE80', 'bg': '#082A14', 'emoji': '😊', 'valence':  1.0, 'arousal':  0.6},
    'Neutral':  {'color': '#94A3B8', 'bg': '#111827', 'emoji': '😐', 'valence':  0.0, 'arousal':  0.0},
    'Sad':      {'color': '#60A5FA', 'bg': '#081528', 'emoji': '😢', 'valence': -0.6, 'arousal': -0.4},
    'Surprise': {'color': '#FBBF24', 'bg': '#1E1808', 'emoji': '😲', 'valence':  0.3, 'arousal':  0.9},
}

# ─── Operational Constants ──────────────────────────────────
SMOOTHING_WINDOW: Final = 5      # frames for temporal smoothing
MAX_HISTORY:      Final = 300    # max prediction records in session
GAME_COUNTDOWN:   Final = 10     # seconds per game round


# ─── Helper Functions ───────────────────────────────────────

def positivity_score(probs: list[float]) -> float:
    """Returns −1.0 (most negative) to +1.0 (most positive).

    Args:
        probs: List of 7 probabilities in EMOTIONS order.

    Returns:
        Float between -1.0 and +1.0.
    """
    weights = [-1.0, -0.8, -0.7, 1.0, 0.0, -0.6, 0.3]
    return float(sum(p * w for p, w in zip(probs, weights)))


def emotion_index(emotion: str) -> int:
    """Return the index of an emotion in EMOTIONS list."""
    return EMOTIONS.index(emotion)


# ─── Mood Music Sync Config ─────────────────────────────────
MOOD_MUSIC: Final = {
    'Happy':    {'genre': 'Pop / Upbeat',   'query': 'happy upbeat pop playlist 2024'},
    'Sad':      {'genre': 'Melancholic',     'query': 'melancholic indie sad songs'},
    'Angry':    {'genre': 'Metal / Intense', 'query': 'intense metal workout playlist'},
    'Fear':     {'genre': 'Ambient',         'query': 'calm ambient relaxation music'},
    'Surprise': {'genre': 'Electronic',      'query': 'energetic electronic surprise beats'},
    'Disgust':  {'genre': 'Blues',           'query': 'blues soul emotional music'},
    'Neutral':  {'genre': 'Lo-fi',           'query': 'lofi hip hop study beats'},
}

# ─── Plotly Dark Theme (import in every page that uses Plotly) ──
PLOTLY_LAYOUT: Final = dict(
    paper_bgcolor='#161B22',
    plot_bgcolor='#0D1117',
    font=dict(color='#E6EDF3', family='Space Grotesk, Inter, sans-serif', size=12),
    margin=dict(l=16, r=16, t=32, b=16),
    colorway=['#00D4AA', '#60A5FA', '#4ADE80', '#FBBF24', '#FB923C', '#A855F7', '#FF6B6B'],
    xaxis=dict(gridcolor='rgba(139,148,158,0.12)', linecolor='rgba(139,148,158,0.2)', zerolinecolor='rgba(139,148,158,0.12)'),
    yaxis=dict(gridcolor='rgba(139,148,158,0.12)', linecolor='rgba(139,148,158,0.2)', zerolinecolor='rgba(139,148,158,0.12)'),
)

# ─── Game Badges ────────────────────────────────────────────
BADGES: Final = {
    '😊 Smile Master':  lambda s: s.get('Happy_count', 0) >= 10,
    '🎭 Method Actor':  lambda s: s.get('perfect_rounds', 0) >= 5,
    '🔥 On Fire':       lambda s: s.get('max_streak', 0) >= 5,
    '😐 Poker Face':    lambda s: s.get('Neutral_count', 0) >= 10,
    '😠 Drama Queen':   lambda s: s.get('Angry_count', 0) >= 5,
    '🏆 Grand Master':  lambda s: s.get('total_score', 0) >= 200,
    '⚡ Speed Demon':   lambda s: s.get('speed_bonus_total', 0) >= 100,
}
