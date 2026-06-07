"""
EmotionLens 🎭 — Real-time facial emotion intelligence powered by CNN
Multi-page Streamlit dashboard for face emotion detection.
"""

import streamlit as st
from utils.session_utils import init_session_state
from utils.model_utils import is_model_available, ensure_model_on_cloud

# Page configuration — MUST be the first Streamlit command
st.set_page_config(
    page_title="EmotionLens 🎭",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

# Ensure model is available (attempts auto-download on Streamlit Cloud)
ensure_model_on_cloud()

# Inject custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0;">🎭</h1>
        <h2 style="color: #00D4AA; font-size: 1.3rem; margin: 0;">EmotionLens</h2>
        <p style="color: #8B949E; font-size: 0.8rem; margin-top: 0.25rem;">
            Facial Emotion Intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Model availability indicator
model_available = is_model_available()
if model_available:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.warning("⚠️ No model found")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "🎥 Live Camera",
        "🖼️ Image Analysis",
        "📊 Analytics",
        "🏋️ Train Model",
        "🧠 Model Inspector",
        "🎯 Emotion Game",
        "📖 About",
    ],
    label_visibility="collapsed",
    index=0,
)

st.sidebar.markdown("---")

# Session stats in sidebar
st.sidebar.markdown("### 📊 Session Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Predictions", st.session_state.get('total_predictions', 0))
with col2:
    from utils.session_utils import format_session_duration
    st.metric("Duration", format_session_duration())

# Dark mode toggle
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.get('dark_mode', True))
if dark_mode != st.session_state.get('dark_mode'):
    st.session_state.dark_mode = dark_mode
    st.rerun()

# ─── API Server Info ─────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("🌐 API Server", expanded=False):
    st.markdown(
        """
        **REST API** for external apps to consume the emotion model.
        
        ```bash
        # Start the server
        python api_server.py
        ```
        
        **Endpoints:**
        - `POST /predict` — base64 image → emotion JSON
        - `POST /predict-file` — upload image file → emotion JSON
        - `GET  /health` — server status
        - `GET  /docs` — Swagger UI
        
        **Quick test:**
        ```bash
        curl -X POST http://localhost:8000/predict \\
          -H "Content-Type: application/json" \\
          -d '{"image": "$(base64 face.jpg)"}'
        ```
        """
    )
    if st.button("🔄 Check API Health", use_container_width=True):
        import requests
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "healthy":
                    st.success("✅ API is running")
                else:
                    st.warning(f"⚠️ API unhealthy: {data}")
            else:
                st.error(f"❌ API returned {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect. Start server: `python api_server.py`")
        except ImportError:
            st.warning("Install `requests` to enable health check.")

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0; color: #8B949E; font-size: 0.7rem;">
        EmotionLens 🎭 v1.0<br>
        Powered by CNN • FER2013
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Page Router ──────────────────────────────────────────────
if page == "🎥 Live Camera":
    from pages.page1_live_camera import show
    show()
elif page == "🖼️ Image Analysis":
    from pages.page2_image_analysis import show
    show()
elif page == "📊 Analytics":
    from pages.page3_analytics import show
    show()
elif page == "🏋️ Train Model":
    from pages.page4_train_model import show
    show()
elif page == "🧠 Model Inspector":
    from pages.page5_model_inspector import show
    show()
elif page == "🎯 Emotion Game":
    from pages.page6_emotion_game import show
    show()
elif page == "📖 About":
    from pages.page7_about import show
    show()
