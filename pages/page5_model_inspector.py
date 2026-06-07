"""
Page 5: 🧠 Model Inspector — CNN architecture visualizer
Displays layer-by-layer breakdown, feature maps, and Grad-CAM heatmaps.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from utils.model_utils import (
    load_model_cached, load_face_cascade, get_model_summary,
    EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME,
)
from utils.emotion_utils import predict_emotion, preprocess_face, compute_gradcam

# Set matplotlib dark theme
plt.style.use('dark_background')


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>🧠 Model Inspector</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Explore the CNN architecture, visualize feature maps, and Grad-CAM heatmaps
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model_cached()
    if model is None:
        return

    # ─── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Architecture", "📊 Parameters", "🎨 Feature Maps", "🔥 Grad-CAM"
    ])

    with tab1:
        _render_architecture_tab(model)
    
    with tab2:
        _render_parameters_tab(model)
    
    with tab3:
        _render_feature_maps_tab(model)
    
    with tab4:
        _render_gradcam_tab(model)


def _render_architecture_tab(model):
    """Display layer-by-layer architecture breakdown."""
    st.markdown("### 📋 Layer-by-Layer Architecture")
    
    layers_info, _ = get_model_summary(model)
    
    # Create a visual table using styled HTML
    rows_html = ""
    for i, layer in enumerate(layers_info):
        row_color = "#1C2128" if i % 2 == 0 else "#161B22"
        rows_html += f"""
        <tr style="background: {row_color};">
            <td style="padding: 8px 12px; border-bottom: 1px solid #30363D; color: #E6EDF3;">{i + 1}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #30363D; color: #00D4AA; font-family: monospace;">{layer['name']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #30363D; color: #F5A623; font-family: monospace;">{layer['type']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #30363D; color: #8B949E; font-family: monospace;">{layer['output_shape']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #30363D; color: '#E6EDF3'; text-align: right; font-family: monospace;">{layer['params']:,}</td>
        </tr>
        """
    
    st.markdown(
        f"""
        <div style="overflow-x: auto; border: 1px solid #30363D; border-radius: 8px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #0D1117; border-bottom: 2px solid #30363D;">
                        <th style="padding: 10px 12px; text-align: left; color: #8B949E; font-weight: 600;">#</th>
                        <th style="padding: 10px 12px; text-align: left; color: #8B949E; font-weight: 600;">Name</th>
                        <th style="padding: 10px 12px; text-align: left; color: #8B949E; font-weight: 600;">Type</th>
                        <th style="padding: 10px 12px; text-align: left; color: #8B949E; font-weight: 600;">Output Shape</th>
                        <th style="padding: 10px 12px; text-align: right; color: #8B949E; font-weight: 600;">Params</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Get model summary text
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
    
    with st.expander("📄 Full Model Summary (Keras)"):
        st.code(summary_buffer.getvalue())
    
    # Architecture flow diagram
    st.markdown("### 🔄 Architecture Flow Diagram")
    _render_flow_diagram(model)


def _render_flow_diagram(model):
    """Render a visual flow diagram of the model architecture."""
    
    layers_info, _ = get_model_summary(model)
    
    # Create boxes for each layer
    fig = go.Figure()
    
    y_pos = 0
    for i, layer in enumerate(layers_info):
        layer_type = layer['type']
        
        # Determine shape and color based on layer type
        if 'Conv2D' in layer_type:
            box_color = '#2ECC71'
            width = 0.8
        elif 'Pooling' in layer_type:
            box_color = '#3498DB'
            width = 0.6
        elif 'Dropout' in layer_type:
            box_color = '#F39C12'
            width = 0.4
        elif 'Flatten' in layer_type:
            box_color = '#9B59B6'
            width = 0.5
        elif 'Dense' in layer_type:
            box_color = '#E74C3C'
            width = 0.7
        else:
            box_color = '#95A5A6'
            width = 0.5
        
        # Draw box
        fig.add_shape(
            type="rect",
            x0=-width/2, x1=width/2,
            y0=y_pos - 0.3, y1=y_pos + 0.3,
            line=dict(color=box_color, width=2),
            fillcolor=box_color,
            opacity=0.3,
        )
        
        # Label
        label = layer['name'].replace('_', ' ')
        fig.add_annotation(
            x=0, y=y_pos,
            text=f"<b>{layer_type}</b><br><span style='font-size:10px'>{label}</span>",
            showarrow=False,
            font=dict(size=11, color='#E6EDF3'),
            align='center',
        )
        
        # Connector line (except for last)
        if i < len(layers_info) - 1:
            fig.add_shape(
                type="line",
                x0=0, y0=y_pos - 0.3,
                x1=0, y1=y_pos - 1.3 + 0.3,
                line=dict(color="#30363D", width=2, dash="dot"),
            )
        
        y_pos -= 1.3
    
    fig.update_layout(
        height=max(300, len(layers_info) * 40),
        xaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[y_pos, 0.5], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0D1117',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_parameters_tab(model):
    """Display parameter count metrics and visualization."""
    st.markdown("### 📊 Parameter Count")
    
    layers_info, param_counts = get_model_summary(model)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div class="badge-total">{param_counts['total']:,}</div>
                <p style="color: #8B949E; margin-top: 0.5rem;">Total Parameters</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div class="badge-trainable">{param_counts['trainable']:,}</div>
                <p style="color: #8B949E; margin-top: 0.5rem;">Trainable Parameters</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div class="badge-non-trainable">{param_counts['non_trainable']:,}</div>
                <p style="color: #8B949E; margin-top: 0.5rem;">Non-Trainable Parameters</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Layer-wise parameter distribution
    st.markdown("### 📈 Parameter Distribution by Layer")
    
    layer_names = [l['name'][:20] for l in layers_info]
    layer_params = [l['params'] for l in layers_info]
    layer_types = [l['type'] for l in layers_info]
    
    colors = []
    for lt in layer_types:
        if 'Conv2D' in lt:
            colors.append('#2ECC71')
        elif 'Dense' in lt:
            colors.append('#E74C3C')
        else:
            colors.append('#3498DB')
    
    fig = go.Figure(data=[go.Bar(
        x=layer_names,
        y=layer_params,
        marker_color=colors,
        text=[f"{p:,}" for p in layer_params],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Type: %{customdata}<br>Params: %{y:,}<extra></extra>',
        customdata=layer_types,
    )])
    
    fig.update_layout(
        yaxis=dict(title="Parameters", type="log", gridcolor='#30363D'),
        xaxis=dict(title="Layer", tickangle=45, tickfont=dict(size=10)),
        height=400,
        margin=dict(l=20, r=20, t=20, b=100),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart of parameter types
    st.markdown("### 🧩 Parameter Type Distribution")
    
    conv_params = sum(l['params'] for l in layers_info if 'Conv2D' in l['type'])
    dense_params = sum(l['params'] for l in layers_info if 'Dense' in l['type'])
    other_params = param_counts['total'] - conv_params - dense_params
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Convolutional', 'Dense', 'Other'],
        values=[conv_params, dense_params, other_params],
        marker=dict(colors=['#2ECC71', '#E74C3C', '#3498DB']),
        textinfo='label+percent',
        hole=0.4,
    )])
    
    fig_pie.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)


def _render_feature_maps_tab(model):
    """Visualize feature maps from intermediate convolutional layers."""
    st.markdown("### 🎨 Feature Map Visualizer")
    
    st.info(
        "Upload an image to visualize the activation maps from each convolutional layer. "
        "This shows what patterns each filter is detecting."
    )
    
    uploaded_file = st.file_uploader(
        "📁 Upload a face image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a face image to see how each convolutional layer processes it.",
    )
    
    if uploaded_file is None:
        return
    
    # Read image
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    img_array = np.array(pil_image.convert('L'))  # Grayscale
    img_resized = cv2.resize(img_array, (48, 48))
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=[0, -1])  # (1, 48, 48, 1)
    
    # Get conv layers
    conv_layers = [
        (i, layer) for i, layer in enumerate(model.layers) 
        if 'conv2d' in layer.name.lower()
    ]
    
    if not conv_layers:
        st.warning("No convolutional layers found in this model.")
        return
    
    # Layer selector
    layer_options = [f"Layer {i}: {l.name} (filters: {l.filters})" for i, l in conv_layers]
    selected_layer_idx = st.select_slider(
        "Select convolutional layer to visualize",
        options=list(range(len(layer_options))),
        format_func=lambda x: layer_options[x],
        value=0,
    )
    
    # Get the selected conv layer
    layer_idx, conv_layer = conv_layers[selected_layer_idx]
    
    # Create a submodel that outputs the activations of the selected conv layer
    from tensorflow.keras.models import Model as KerasModel
    
    activation_model = KerasModel(inputs=model.input, outputs=model.layers[layer_idx].output)
    activations = activation_model.predict(img_input, verbose=0)[0]
    
    st.markdown(f"#### Feature Maps for Layer: {conv_layer.name}")
    st.markdown(f"Shape: {activations.shape} — {activations.shape[-1]} filters of size {activations.shape[0]}×{activations.shape[1]}")
    
    # Display original image
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(pil_image, caption="Original Image", use_container_width=True)
    
    # Display feature maps in a grid
    n_filters = activations.shape[-1]
    cols = min(8, n_filters)
    rows = (n_filters + cols - 1) // cols
    max_maps = min(n_filters, 48)  # Max 48 filters to display
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for i in range(max_maps):
        ax = axes[i]
        ax.imshow(activations[:, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'F{i}', fontsize=8, color='white')
    
    # Hide unused subplots
    for i in range(max_maps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_gradcam_tab(model):
    """Generate Grad-CAM heatmaps for visual explanations."""
    st.markdown("### 🔥 Grad-CAM Heatmap")
    
    st.info(
        "Grad-CAM highlights the regions of the image that the model focuses on "
        "when making its prediction. Upload a face image to see the heatmap."
    )
    
    uploaded_file = st.file_uploader(
        "📁 Upload a face image for Grad-CAM",
        type=['jpg', 'jpeg', 'png'],
        key="gradcam_upload",
        help="Upload a face image to generate Grad-CAM heatmap overlay.",
    )
    
    if uploaded_file is None:
        return
    
    # Process image
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    img_array = np.array(pil_image.convert('L'))
    img_resized = cv2.resize(img_array, (48, 48))
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=[0, -1])  # (1, 48, 48, 1)
    
    # Make prediction first
    from utils.emotion_utils import predict_emotion, compute_gradcam
    # Use grayscale face ROI directly
    emotion, confidence, probs = predict_emotion(model, img_resized)
    
    st.markdown(f"**Prediction:** {EMOTION_CONFIG.get(emotion, {}).get('emoji', '')} {emotion} ({confidence*100:.1f}%)")
    
    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            # Get target class index
            target_class = int(np.argmax(probs))
            
            # Compute Grad-CAM using the shared utility function
            heatmap = compute_gradcam(model, img_input, target_class)
            
            if heatmap is None:
                st.error("Grad-CAM computation failed. No convolutional layers found.")
                return
            
            # Resize heatmap to original image size
            heatmap_resized = cv2.resize(heatmap, pil_image.size)
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Convert PIL to numpy for overlay
            pil_rgb = np.array(pil_image.convert('RGB'))
            overlay = cv2.addWeighted(pil_rgb, 0.6, heatmap_colored, 0.4, 0)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(pil_image, caption="Original Image", use_container_width=True)
            
            with col2:
                st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_container_width=True)
            
            with col3:
                st.image(overlay, caption="Overlay", use_container_width=True)
            
            # Explanation
            st.markdown(
                f"""
                <div style="
                    background: #1C2128;
                    border: 1px solid #30363D;
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-top: 1rem;
                ">
                    <h4 style="color: #00D4AA; margin: 0 0 0.5rem 0;">🔍 Interpretation</h4>
                    <p style="color: #E6EDF3; margin: 0;">
                        The red/orange regions show where the CNN focused most for predicting 
                        <strong>{EMOTION_CONFIG.get(emotion, {}).get('emoji', '')} {emotion}</strong>. 
                        Look at the mouth, eyes, and brow areas — these facial features are most 
                        important for emotion classification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        except Exception as e:
            st.error(f"❌ Grad-CAM failed: {e}")
            st.exception(e)
