# Author: Upenyu Hlangabeza
"""Explainability page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.services.explainability_service import global_importance, local_contributions
from app.services.prediction_service import applicant_to_frame
from app.services.preprocessing_service import clean_for_modeling


def render() -> None:
    """Render global and local explainability views."""
    page_header(
        "Explainability",
        "Why the Model Predicts Risk",
        "Use global importance and local contribution analysis to understand model decisions.",
    )

    active = st.session_state.training_results.get(st.session_state.active_model_name)
    if active is None:
        st.info("Train a model first to unlock explainability.")
        return

    section_title("Global Feature Importance", f"Active model: {active.name}")
    st.plotly_chart(charts.feature_importance(global_importance(active)), use_container_width=True)

    prediction = st.session_state.last_prediction
    if prediction is None:
        st.info("Run a prediction to view individual applicant explanations. Showing the first portfolio row as a sample.")
        sample = clean_for_modeling(st.session_state.raw_df).iloc[0]
        explanation = local_contributions(active, st.session_state.raw_df, sample)
    else:
        sample = applicant_to_frame(prediction.applicant).iloc[0]
        explanation = prediction.explanation

    left, right = st.columns([1, 1])
    with left:
        section_title("Local Contribution Chart", "Positive values push risk higher; negative values reduce risk.")
        st.plotly_chart(charts.contribution_chart(explanation), use_container_width=True)
    with right:
        section_title("Contribution Table", "Baseline replacement analysis for the selected applicant.")
        st.dataframe(explanation, use_container_width=True, height=420)

    with st.expander("Method", expanded=False):
        st.write(
            "This page uses model-agnostic perturbation analysis. Each feature is replaced with a portfolio "
            "baseline value and the change in predicted default probability is measured. It is not a full SHAP "
            "implementation, but it gives an intuitive local explanation without adding a heavy dependency."
        )
