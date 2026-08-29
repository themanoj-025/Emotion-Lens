"""Analytics dashboard -- main show() function."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from pages.analytics_pkg.charts import (
    _render_confidence_chart,
    _render_distribution_chart,
    _render_heatmap,
    _render_positivity_analysis,
    _render_timeline_chart,
)
from pages.analytics_pkg.timeline import _render_timeline_replay
from utils.model_utils import EMOTION_CONFIG
from utils.session_utils import (
    export_predictions_csv,
    export_predictions_json,
    format_session_duration,
    get_prediction_dataframe,
    reset_session,
)


def show() -> None:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <h1>📊 Analytics Dashboard</h1>
            <p style="color: #8B949E; font-size: 1rem;">
                Session statistics, emotion distribution, and trends
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_prediction_dataframe()

    # Top Metrics
    st.markdown("### 📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    total_preds = st.session_state.get("total_predictions", 0)

    with col1:
        st.metric("Total Predictions", total_preds)

    with col2:
        if df.empty:
            st.metric("Session Duration", "00:00:00")
        else:
            st.metric("Session Duration", format_session_duration())

    with col3:
        if not df.empty:
            most_common = df["emotion"].mode()
            top_emotion = most_common.iloc[0] if not most_common.empty else "—"
            emoji = EMOTION_CONFIG.get(top_emotion, {}).get("emoji", "")
            st.metric("Most Detected", f"{emoji} {top_emotion}")
        else:
            st.metric("Most Detected", "—")

    with col4:
        if not df.empty:
            avg_conf = df["confidence"].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        else:
            st.metric("Avg Confidence", "—%")

    if df.empty:
        st.info(
            "📭 No predictions recorded yet. Use the **Live Camera** or **Image Analysis** pages to collect data."
        )
        return

    # Charts
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🎯 Distribution",
            "📈 Over Time",
            "📊 Confidence",
            "🔥 Heatmap",
            "💚 Positivity",
            "🎬 Timeline Replay",
        ]
    )

    with tab1:
        _render_distribution_chart(df)

    with tab2:
        _render_timeline_chart(df)

    with tab3:
        _render_confidence_chart(df)

    with tab4:
        _render_heatmap(df)

    with tab5:
        _render_positivity_analysis(df)

    with tab6:
        _render_timeline_replay(df)

    # Raw Data & Export
    st.markdown("---")
    st.markdown("### 📋 Raw Data & Export")

    with st.expander("📄 View Raw Prediction Data"):
        display_df = df.drop(columns=["probabilities"], errors="ignore")
        st.dataframe(display_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_predictions_csv()
        if csv_data:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

    with col2:
        json_data = export_predictions_json()
        if json_data:
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

    with col3:
        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()


