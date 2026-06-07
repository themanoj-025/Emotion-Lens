"""
Page 7: 📖 About — Project information, dataset details, tech stack, and documentation
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.model_utils import EMOTIONS, EMOTION_CONFIG, PLOTLY_THEME, is_model_available, load_model_cached


def show():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>📖 About EmotionLens 🎭</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Real-time facial emotion intelligence powered by CNN
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Overview ─────────────────────────────────────────────
    st.markdown("## 🎯 Project Overview")
    
    st.markdown(
        """
        <div style="
            background: #1C2128;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 1.5rem;
            line-height: 1.8;
        ">
        <p>
            <strong>EmotionLens</strong> is a production-grade deep learning application that 
            detects and classifies facial expressions into <strong>7 emotions</strong>:
            😠 Angry, 🤢 Disgust, 😨 Fear, 😊 Happy, 😐 Neutral, 😢 Sad, and 😲 Surprise.
        </p>
        <p>
            Powered by a <strong>Convolutional Neural Network (CNN)</strong> trained on the 
            <strong>FER2013 dataset</strong>, the system achieves approximately <strong>62% validation accuracy</strong>.
        </p>
        <p>
            This application features a multi-page Streamlit dashboard with real-time webcam 
            detection, static image analysis, interactive games, model training, and more.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Model Architecture ───────────────────────────────────
    st.markdown("## 🧠 Model Architecture")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(
            """
            The model is a standard CNN with **3 convolutional blocks**:
            
            1. **Input**: 48×48 grayscale image (1 channel)
            2. **Conv Block 1**: Conv2D (32 filters) → MaxPooling → Dropout (0.25)
            3. **Conv Block 2**: Conv2D (64 filters) → MaxPooling → Dropout (0.25)
            4. **Conv Block 3**: Conv2D (128 filters) → MaxPooling → Dropout (0.25)
            5. **Flatten**: Convert 2D features to 1D vector
            6. **Dense**: 1024 units with ReLU activation → Dropout (0.5)
            7. **Output**: 7 units with Softmax activation
            
            - **Optimizer**: Adam
            - **Loss**: Categorical Crossentropy
            - **Total Parameters**: ~1.2M
            """
        )
    
    with col2:
        # Simple architecture visualization
        layers = [
            ('Input', '48×48×1', '#3498DB'),
            ('Conv2D', '32', '#2ECC71'),
            ('MaxPool', '24×24', '#3498DB'),
            ('Conv2D', '64', '#2ECC71'),
            ('MaxPool', '12×12', '#3498DB'),
            ('Conv2D', '128', '#2ECC71'),
            ('MaxPool', '6×6', '#3498DB'),
            ('Flatten', '4608', '#9B59B6'),
            ('Dense', '1024', '#E74C3C'),
            ('Output', '7', '#F5A623'),
        ]
        
        fig = go.Figure()
        for i, (name, dim, color) in enumerate(layers):
            fig.add_trace(go.Bar(
                x=[1],
                y=[1],
                name=f"{name} ({dim})",
                marker_color=color,
                text=f"<b>{name}</b><br>{dim}",
                textposition='inside',
                hovertemplate=f"<b>{name}</b><br>Dimension: {dim}<extra></extra>",
            ))
        
        fig.update_layout(
            barmode='stack',
            height=400,
            showlegend=True,
            legend=dict(font=dict(size=10, color='#8B949E')),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=10),
            **PLOTLY_THEME,
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # ─── FER2013 Dataset ──────────────────────────────────────
    st.markdown("## 📊 FER2013 Dataset")
    
    st.markdown(
        """
        The **FER2013** (Facial Expression Recognition 2013) dataset was introduced in the 
        ICML 2013 workshop on Challenges in Representation Learning. It consists of:
        
        - **35,887** grayscale images of faces
        - **48×48** pixels resolution
        - **7 emotion categories**
        - **28,709** training examples
        - **3,589** validation/test examples per split
        """
    )
    
    # Class distribution chart
    st.markdown("### 📈 Class Distribution")
    
    # Approximate FER2013 training distribution
    distribution = {
        'Angry': 3995,
        'Disgust': 436,
        'Fear': 4097,
        'Happy': 7215,
        'Neutral': 4965,
        'Sad': 4830,
        'Surprise': 3171,
    }
    
    colors = [EMOTION_CONFIG.get(e, {}).get('color', '#95A5A6') for e in distribution.keys()]
    emoji_labels = [f"{EMOTION_CONFIG.get(e, {}).get('emoji', '')} {e}" for e in distribution.keys()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emoji_labels,
            y=list(distribution.values()),
            marker_color=colors,
            text=[f"{v:,} ({v/28709*100:.1f}%)" for v in distribution.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[v/28709*100 for v in distribution.values()],
        )
    ])
    
    fig.update_layout(
        title="FER2013 Training Set Class Distribution",
        yaxis=dict(title="Sample Count", gridcolor='#30363D'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        **PLOTLY_THEME,
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ─── Performance ──────────────────────────────────────────
    st.markdown("## 📉 Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Validation Accuracy", "~62%", delta="Standard for FER2013 CNNs")
    
    with col2:
        if is_model_available():
            model = load_model_cached()
            if model:
                st.metric("Model Size (params)", f"{model.count_params():,}")
    
    st.markdown(
        """
        <div style="
            background: #1C2128;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <p style="color: #8B949E; margin: 0;">
                💡 <strong>Note:</strong> 62% validation accuracy is considered good for the 
                FER2013 dataset with a simple CNN. The dataset contains challenging, 
                real-world images with varied lighting, pose, and occlusion. State-of-the-art 
                models achieve ~73% using ensembles and attention mechanisms.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Tech Stack ──────────────────────────────────────────
    st.markdown("## 🛠️ Tech Stack")
    
    tech_stack = {
        "Python": "3.8+",
        "TensorFlow": "Deep learning framework",
        "Keras": "High-level neural network API",
        "OpenCV": "Computer vision & image processing",
        "Streamlit": "Dashboard & UI framework",
        "Plotly": "Interactive visualizations",
        "NumPy": "Numerical computing",
        "Pandas": "Data manipulation & analysis",
        "Matplotlib": "Static plot generation",
        "KaggleHub": "Dataset download",
    }
    
    st.table(pd.DataFrame([
        {"Technology": tech, "Description": desc}
        for tech, desc in tech_stack.items()
    ]))

    # ─── GitHub README ────────────────────────────────────────
    st.markdown("## 📖 README")
    
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            readme_content = f.read()
        
        with st.expander("📄 View Project README"):
            st.markdown(readme_content)
    else:
        st.info("README.md not found in the project root.")

    # ─── Author & License ─────────────────────────────────────
    st.markdown("---")
    
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 2rem;
            color: #8B949E;
        ">
            <h3 style="color: #00D4AA; margin-bottom: 1rem;">EmotionLens 🎭</h3>
            <p>Built with ❤️ using TensorFlow, Streamlit, and OpenCV</p>
            <p style="font-size: 0.9rem;">
                FER2013 dataset courtesy of ICML 2013 Workshop on Challenges in Representation Learning<br>
                Model architecture based on standard CNN for facial expression recognition
            </p>
            <div style="margin-top: 1rem;">
                <span style="background: #1C2128; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363D;">
                    🐍 Python
                </span>
                <span style="background: #1C2128; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363D; margin-left: 0.5rem;">
                    🧠 TensorFlow
                </span>
                <span style="background: #1C2128; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363D; margin-left: 0.5rem;">
                    📊 Streamlit
                </span>
                <span style="background: #1C2128; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363D; margin-left: 0.5rem;">
                    👁️ OpenCV
                </span>
            </div>
            <p style="margin-top: 2rem; font-size: 0.8rem;">
                © 2026 EmotionLens Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



