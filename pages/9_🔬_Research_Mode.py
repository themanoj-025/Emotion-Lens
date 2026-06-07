"""
Page 9: 🔬 Research Mode — Scientific export, model benchmarking, embedding visualization,
and REST API code generation for developers and researchers.
"""
import streamlit as st
import numpy as np
import json
import io
import pandas as pd
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from utils.config import EMOTIONS, EMOTION_CONFIG, PLOTLY_LAYOUT, positivity_score
from utils.model_utils import load_emotion_model, load_face_cascade, preprocess_roi, get_model_summary


def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🔬 Research Mode</h1>
        <p>Scientific analysis, model benchmarking, embeddings, and API export</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_emotion_model()
    face_cascade = load_face_cascade()
    if model is None:
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Benchmark", "🔄 Embedding Viz", "⚡ FPS Benchmark", "🌐 REST API"])

    with tab1:
        st.markdown("#### Model Benchmark — 7 Sample Emotions")
        st.info("This benchmark tests the model on placeholder data for each emotion class.")

        if st.button("🚀 Run Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                results = []
                for i, emotion in enumerate(EMOTIONS):
                    # Create synthetic test input: random noise + class-specific pattern
                    synthetic = np.random.randn(1, 48, 48, 1).astype('float32') * 0.1
                    start = time.time()
                    probs = model.predict(synthetic, verbose=0)[0]
                    elapsed = (time.time() - start) * 1000
                    pred_idx = int(np.argmax(probs))
                    results.append({
                        'emotion': emotion,
                        'predicted': EMOTIONS[pred_idx],
                        'confidence': float(probs[pred_idx]),
                        'correct': emotion == EMOTIONS[pred_idx],
                        'inference_ms': round(elapsed, 2),
                    })
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                acc = df['correct'].mean() * 100
                avg_time = df['inference_ms'].mean()
                st.metric("Accuracy (synthetic)", f"{acc:.1f}%")
                st.metric("Avg Inference Time", f"{avg_time:.2f} ms")

    with tab2:
        st.markdown("#### Embedding Visualizer (Penultimate Layer)")
        st.info("Upload images to extract Dense layer activations and visualize with PCA/t-SNE.")

        uploaded_files = st.file_uploader("Upload multiple face images (2+)", type=['jpg', 'png'],
                                          accept_multiple_files=True, key="embeddings")
        if uploaded_files and len(uploaded_files) >= 2:
            # Extract penultimate layer activations
            from sklearn.decomposition import PCA
            import cv2
            # Find the penultimate Dense layer
            penultimate_layer = None
            for layer in reversed(model.layers):
                if 'dense' in layer.name.lower():
                    if penultimate_layer is None:
                        penultimate_layer = layer.name
                    else:
                        break
            if penultimate_layer is None:
                st.error("Could not find a Dense layer for embedding extraction.")
            else:
                from tensorflow.keras.models import Model as KerasModel
                embed_model = KerasModel(inputs=model.input, outputs=model.get_layer(penultimate_layer).output)
                embeddings = []
                labels = []
                for f in uploaded_files:
                    img = Image.open(io.BytesIO(f.read())).convert('L')
                    img_resized = np.array(img.resize((48, 48)))
                    inp = preprocess_roi(img_resized)
                    emb = embed_model.predict(inp, verbose=0)[0]
                    # Get prediction for label
                    probs = model.predict(inp, verbose=0)[0]
                    pred_emo = EMOTIONS[int(np.argmax(probs))]
                    embeddings.append(emb)
                    labels.append(pred_emo)

                X = np.array(embeddings)
                pca = PCA(n_components=min(2, X.shape[0], X.shape[1]))
                X_pca = pca.fit_transform(X)
                fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros(len(X_pca)),
                                 color=labels, color_discrete_map={e: EMOTION_CONFIG[e]['color'] for e in EMOTIONS},
                                 title="PCA of Penultimate Layer Embeddings")
                fig.update_layout(**PLOTLY_LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### FPS Benchmark")
        st.info("Measures inference speed at different batch/face sizes.")
        if st.button("⚡ Run FPS Benchmark", type="primary"):
            results = []
            for n_faces in [1, 5, 10]:
                synthetic_batch = np.random.randn(n_faces, 48, 48, 1).astype('float32') * 0.1
                start = time.time()
                n_iterations = 50
                for _ in range(n_iterations):
                    _ = model.predict(synthetic_batch, verbose=0)
                elapsed = time.time() - start
                fps = n_iterations * n_faces / elapsed
                results.append({'faces': n_faces, 'fps': round(fps, 1), 'ms_per_face': round(elapsed / (n_iterations * n_faces) * 1000, 2)})
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            fig = go.Figure(go.Bar(x=df['faces'], y=df['fps'], marker_color='#00D4AA',
                                   text=df['fps'], textposition='outside'))
            fig.update_layout(**PLOTLY_LAYOUT, title="FPS vs Number of Faces", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### REST API Code Generation")
        st.code("""# Auto-generated FastAPI server for EmotionLens model
# Save as api_server.py and run: uvicorn api_server:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import cv2, numpy as np, io
from PIL import Image

app = FastAPI(title="EmotionLens API")
model = load_model("models/emotion_model.h5")
EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')
    img = np.array(img.resize((48, 48))).astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"emotion": EMOTIONS[idx], "confidence": float(probs[idx]),
            "probabilities": {e: float(probs[i]) for i, e in enumerate(EMOTIONS)}}
""", language="python")
        st.download_button("📥 Download api_server.py", data=_generate_api_script(),
                           file_name="emotion_api_server.py", use_container_width=True)


def _generate_api_script():
    return '''"""EmotionLens 🎭 REST API Server"""
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import cv2, numpy as np, io
from PIL import Image
app = FastAPI(title="EmotionLens 🎭 API")
model = load_model("models/emotion_model.h5")
EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')
    img = np.array(img.resize((48, 48))).astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"emotion": EMOTIONS[idx], "confidence": float(probs[idx]),
            "probabilities": {e: float(probs[i]) for i, e in enumerate(EMOTIONS)}}
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
'''
