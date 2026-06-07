"""Grad-CAM heatmap generation for CNN emotion model."""
from __future__ import annotations
import numpy as np
import cv2
import tensorflow as tf


def make_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str = None) -> np.ndarray:
    """Generate Grad-CAM heatmap for the top predicted class.

    Args:
        model: Loaded Keras model.
        img_array: Input array of shape (1, 48, 48, 1), normalized [0, 1].
        last_conv_layer_name: Name of last conv layer. Auto-detected if None.

    Returns:
        Heatmap ndarray of shape (48, 48), values in [0, 1].

    Raises:
        ValueError: If no convolutional layer is found.
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model.")

    # Build gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        top_class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, top_class_idx]

    grads = tape.gradient(class_score, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(original_img_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original face image. Returns BGR image.

    Args:
        original_img_gray: Grayscale face image.
        heatmap: Heatmap array from make_gradcam_heatmap().
        alpha: Blending factor (0 = no overlay, 1 = only heatmap).

    Returns:
        BGR image with heatmap overlay.
    """
    h, w = original_img_gray.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype('uint8'), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_img_gray, cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed
