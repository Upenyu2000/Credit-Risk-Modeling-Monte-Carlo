# Author: Upenyu Hlangabeza
"""Model evaluation page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.services.model_service import metrics_frame


def render() -> None:
    """Render interactive evaluation artifacts for trained models."""
    page_header(
        "Model Evaluation",
        "Diagnostics and Model Comparison",
        "Inspect confusion matrices, ROC/PR curves, feature importance, and classification reports.",
    )

    results = st.session_state.training_results
    if not results:
        st.info("Train at least one model from the Model Training page.")
        return

    names = list(results.keys())
    active_name = st.selectbox("Active model", names, index=names.index(st.session_state.active_model_name) if st.session_state.active_model_name in names else 0)
    st.session_state.active_model_name = active_name
    result = results[active_name]

    section_title("Comparison", "Holdout metrics across trained algorithms.")
    st.plotly_chart(charts.model_comparison(metrics_frame(results)), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.confusion_matrix(result.confusion_matrix), use_container_width=True)
        st.plotly_chart(charts.roc_curve_chart(result.y_test, result.y_proba, active_name), use_container_width=True)
    with right:
        st.plotly_chart(charts.precision_recall_chart(result.y_test, result.y_proba, active_name), use_container_width=True)
        st.plotly_chart(charts.feature_importance(result.feature_importance), use_container_width=True)

    section_title("Classification Report", "Precision, recall, F1, and support by class.")
    st.dataframe(result.classification_report, use_container_width=True)
