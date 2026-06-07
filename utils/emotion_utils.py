"""
Emotion detection utilities for EmotionLens 🎭
Handles image preprocessing, prediction logic, and visualization helpers.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

import streamlit as st

from utils.model_utils import EMOTIONS, EMOTION_CONFIG, MOOD_MUSIC_MAP


def preprocess_face(face_roi, target_size=(48, 48)):
    """
    Preprocess a face ROI for model prediction.
    
    Args:
        face_roi: Grayscale face region (numpy array)
        target_size: (width, height) tuple, default (48, 48)
    
    Returns:
        Preprocessed array of shape (1, 48, 48, 1)
    """
    roi_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    roi_array = roi_resized.astype('float32') / 255.0
    roi_array = np.expand_dims(roi_array, axis=-1)  # Add channel dim
    roi_array = np.expand_dims(roi_array, axis=0)   # Add batch dim
    return roi_array


def predict_emotion(model, face_roi):
    """
    Predict emotion from a preprocessed face ROI.
    
    Args:
        model: Loaded Keras model
        face_roi: Grayscale face region (numpy array)
    
    Returns:
        Tuple of (predicted_emotion, confidence, all_probabilities)
    """
    processed = preprocess_face(face_roi)
    predictions = model.predict(processed, verbose=0)[0]
    
    max_idx = int(np.argmax(predictions))
    predicted_emotion = EMOTIONS[max_idx]
    confidence = float(predictions[max_idx])
    
    return predicted_emotion, confidence, predictions


def predict_from_image(model, face_cascade, image, detect_faces=True):
    """
    Predict emotions from a full image, detecting faces first.
    
    Args:
        model: Loaded Keras model
        face_cascade: OpenCV CascadeClassifier
        image: BGR numpy array (OpenCV format) or PIL Image
        detect_faces: If True, auto-detect faces. If False, use full image.
    
    Returns:
        List of dicts with keys: emotion, confidence, probabilities, bbox, face_img
    """
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_array = image.copy()
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    results = []
    
    if detect_faces:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            # Fallback: use the whole image
            results.append(_predict_single_face(model, gray, (0, 0, gray.shape[1], gray.shape[0])))
            results[-1]['fallback'] = True
            return results
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            result = _predict_single_face(model, face_roi, (x, y, w, h))
            if result:
                results.append(result)
    else:
        result = _predict_single_face(model, gray, (0, 0, gray.shape[1], gray.shape[0]))
        if result:
            results.append(result)
    
    return results


def _predict_single_face(model, face_roi, bbox):
    """Helper to predict emotion on a single face ROI."""
    x, y, w, h = bbox
    
    try:
        emotion, confidence, all_probs = predict_emotion(model, face_roi)
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': all_probs.tolist(),
            'bbox': (x, y, w, h),
        }
    except Exception as e:
        return None


def draw_detection_result(image, result):
    """
    Draw emotion detection results on an image (in-place).
    
    Args:
        image: BGR numpy array (will be modified in-place)
        result: Dict from predict_from_image()
    
    Returns:
        Modified image with bounding boxes and labels
    """
    x, y, w, h = result['bbox']
    emotion = result['emotion']
    confidence = result['confidence']
    config = EMOTION_CONFIG.get(emotion, {'color': '#FFFFFF', 'emoji': '❓'})
    
    # Convert hex color to BGR
    hex_color = config['color'].lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, 3)
    
    # Draw label with background
    label = f"{config['emoji']} {emotion} ({confidence*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Label background
    cv2.rectangle(image, (x, y - th - 15), (x + tw + 10, y), bgr_color, -1)
    
    # Label text
    cv2.putText(image, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Confidence bar below face
    bar_width = int(w * confidence)
    bar_y = y + h + 8
    cv2.rectangle(image, (x, bar_y), (x + w, bar_y + 6), (50, 50, 50), -1)
    cv2.rectangle(image, (x, bar_y), (x + bar_width, bar_y + 6), bgr_color, -1)
    
    # Draw confidence percentage text
    cv2.putText(
        image, f"{confidence*100:.0f}%", (x + w + 5, bar_y + 6),
        font, 0.5, (200, 200, 200), 1
    )
    
    return image


def compute_positivity_score(probabilities):
    """
    Compute a positivity/valence score from −1 to +1 based on emotion probabilities.
    
    Formula:
        (Happy*1 + Surprise*0.3 + Neutral*0 - Sad*0.8 - Angry*1 - Fear*0.6 - Disgust*0.5)
    
    Args:
        probabilities: List/array of 7 emotion probabilities in EMOTIONS order
    
    Returns:
        Float between -1.0 and 1.0
    """
    weights = {
        'Angry': -1.0,
        'Disgust': -0.5,
        'Fear': -0.6,
        'Happy': 1.0,
        'Neutral': 0.0,
        'Sad': -0.8,
        'Surprise': 0.3,
    }
    
    score = sum(probabilities[i] * weights[emotion] for i, emotion in enumerate(EMOTIONS))
    return np.clip(score, -1.0, 1.0)


def apply_temporal_smoothing(history, new_prediction, window=5):
    """
    Apply rolling average over recent predictions to reduce flickering.
    
    Args:
        history: List of previous prediction dicts
        new_prediction: Dict with 'probabilities' key
        window: Number of frames to average over
    
    Returns:
        Smoothed prediction dict
    """
    history.append(new_prediction)
    if len(history) > window:
        history.pop(0)
    
    if len(history) < 3:
        return new_prediction
    
    avg_probs = np.mean([h['probabilities'] for h in history], axis=0)
    max_idx = int(np.argmax(avg_probs))
    
    return {
        'emotion': EMOTIONS[max_idx],
        'confidence': float(avg_probs[max_idx]),
        'probabilities': avg_probs.tolist(),
    }


def generate_emotion_summary(results):
    """
    Generate a group/summary text when multiple faces are detected.
    
    Args:
        results: List of prediction result dicts
    
    Returns:
        Summary string like "Your group is 60% Happy, 30% Neutral, 10% Surprised 🎉"
    """
    if not results:
        return "No faces detected."
    
    if len(results) == 1:
        r = results[0]
        emoji = EMOTION_CONFIG.get(r['emotion'], {}).get('emoji', '')
        return f"Detected: {emoji} {r['emotion']} ({r['confidence']*100:.1f}%)"
    
    emotion_counts = {}
    for r in results:
        e = r['emotion']
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    
    total = len(results)
    parts = []
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        emoji = EMOTION_CONFIG.get(emotion, {}).get('emoji', '')
        parts.append(f"{emoji} {emotion} {pct:.0f}%")
    
    return f"Your group is {', '.join(parts)}"


def image_to_base64(pil_image):
    """Convert PIL Image to base64 string for download."""
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


# ─── Grad-CAM ─────────────────────────────────────────────────────
# Cache for the last conv layer index and gradient model to avoid rebuilding
_GRADCAM_CACHE = {}

def _get_last_conv_layer_idx(model):
    """Find the index of the last convolutional layer in the model."""
    last_idx = None
    for i, layer in enumerate(model.layers):
        if 'conv2d' in layer.name.lower():
            last_idx = i
    return last_idx


def _build_grad_model(model):
    """Build and cache the gradient model for Grad-CAM."""
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
    
    model_id = id(model)
    if model_id in _GRADCAM_CACHE:
        return _GRADCAM_CACHE[model_id]
    
    last_conv_idx = _get_last_conv_layer_idx(model)
    if last_conv_idx is None:
        _GRADCAM_CACHE[model_id] = None
        return None
    
    grad_model = KerasModel(
        inputs=model.input,
        outputs=[model.layers[last_conv_idx].output, model.output]
    )
    _GRADCAM_CACHE[model_id] = grad_model
    return grad_model


def compute_gradcam(model, preprocessed_input, target_class_idx):
    """
    Compute Grad-CAM heatmap for a given input and target class.
    
    Args:
        model: Loaded Keras model
        preprocessed_input: Preprocessed image array of shape (1, 48, 48, 1)
        target_class_idx: Index of the target class to explain
    
    Returns:
        2D numpy array (heatmap) of shape (48, 48) with values in [0, 1],
        or None if Grad-CAM fails
    """
    import tensorflow as tf
    
    grad_model = _build_grad_model(model)
    if grad_model is None:
        return None
    
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_input, training=False)
            loss = predictions[:, target_class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        return heatmap
    except Exception:
        return None


# ─── Face Anonymizer ────────────────────────────────────────────

def anonymize_faces(image_bgr, face_cascade=None, kernel_size=(99, 99), pixelate=False):
    """
    Anonymize (blur or pixelate) all detected faces in an image for privacy preservation.
    Applies a strong Gaussian blur to each face region while preserving the rest of the image.
    
    Args:
        image_bgr: BGR numpy array of the image (will be modified in-place)
        face_cascade: OpenCV CascadeClassifier for face detection. If None, uses default.
        kernel_size: Tuple (odd, odd) for Gaussian blur kernel size. Larger = stronger blur.
        pixelate: If True, applies pixelation instead of Gaussian blur (more stylized).
    
    Returns:
        The anonymized BGR image (same array, modified in-place)
    """
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Ensure kernel size is odd and not larger than face
        kx = min(kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1, w)
        ky = min(kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1, h)
        if kx < 3: kx = 3
        if ky < 3: ky = 3
        if kx % 2 == 0: kx += 1
        if ky % 2 == 0: ky += 1
        
        face_roi = image_bgr[y:y+h, x:x+w]
        
        if pixelate:
            # Pixelation: downscale then upscale (more stylized privacy)
            small = cv2.resize(face_roi, (max(4, w // 15), max(4, h // 15)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            image_bgr[y:y+h, x:x+w] = pixelated
        else:
            # Strong Gaussian blur
            blurred = cv2.GaussianBlur(face_roi, (kx, ky), 0)
            image_bgr[y:y+h, x:x+w] = blurred
    
    return image_bgr


# ─── Mood Music Sync ──────────────────────────────────────────

def render_mood_music_card(emotion, confidence=None):
    """
    Render a styled card with Spotify and YouTube search links matching the emotion.
    
    Args:
        emotion: Detected emotion string (must be in EMOTIONS)
        confidence: Optional confidence score (0-1) to display
    """
    music = MOOD_MUSIC_MAP.get(emotion)
    if not music:
        return
    
    config = EMOTION_CONFIG.get(emotion, {})
    bg_color = config.get('bg', '#1C2128')
    accent = config.get('color', '#00D4AA')
    emoji = config.get('emoji', '🎵')
    
    # Build search URLs
    import urllib.parse
    spotify_url = f"https://open.spotify.com/search/{urllib.parse.quote(music['spotify'])}"
    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(music['youtube'])}"
    
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border: 1px solid {accent}44;
            border-radius: 14px;
            padding: 1.2rem 1.2rem 0.8rem;
            margin-top: 0.8rem;
            box-shadow: 0 0 20px {accent}22;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                <span style="font-size: 1.3rem;">🎵</span>
                <span style="color: {accent}; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px;">
                    Mood Music Sync
                </span>
                <span style="margin-left: auto; font-size: 1.2rem;">{emoji} {music['vibe']}</span>
            </div>
            <p style="color: #8B949E; font-size: 0.85rem; margin: 0.2rem 0 0.8rem 0;">
                {music['desc']}
            </p>
            <div style="display: flex; gap: 0.6rem;">
                <a href="{spotify_url}" target="_blank" style="
                    display: inline-flex; align-items: center; gap: 0.4rem;
                    background: #1DB954; color: white; text-decoration: none;
                    padding: 0.45rem 1rem; border-radius: 24px;
                    font-size: 0.85rem; font-weight: 600;
                    transition: transform 0.15s;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    🎧 Spotify
                </a>
                <a href="{youtube_url}" target="_blank" style="
                    display: inline-flex; align-items: center; gap: 0.4rem;
                    background: #FF0000; color: white; text-decoration: none;
                    padding: 0.45rem 1rem; border-radius: 24px;
                    font-size: 0.85rem; font-weight: 600;
                    transition: transform 0.15s;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    ▶️ YouTube
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_gradcam_overlay(frame_bgr, face_bbox, heatmap_small):
    """
    Apply Grad-CAM heatmap overlay on a face region in the frame.
    
    Args:
        frame_bgr: BGR frame (numpy array, modified in-place)
        face_bbox: (x, y, w, h) tuple for the face bounding box
        heatmap_small: 48x48 heatmap array from compute_gradcam()
    
    Returns:
        The modified frame
    """
    x, y, w, h = face_bbox
    
    # Resize heatmap to face bounding box size
    heatmap_resized = cv2.resize(heatmap_small, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Extract face region from frame
    face_region = frame_bgr[y:y+h, x:x+w]
    
    # Blend heatmap with face region
    overlay = cv2.addWeighted(face_region, 0.55, heatmap_colored, 0.45, 0)
    frame_bgr[y:y+h, x:x+w] = overlay
    
    # Add a small legend label
    cv2.putText(
        frame_bgr, "🔥 Grad-CAM", (x, y - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2
    )
    
    return frame_bgr
