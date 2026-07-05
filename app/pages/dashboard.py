# Author: Upenyu Hlangabeza
"""Dashboard page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.config import ALGORITHMS, TARGET_COLUMN
from app.models.schemas import TrainingConfig
from app.services.data_service import summarize_dataset
from app.services.model_service import best_model_name, get_available_algorithms, metrics_frame, train_models
from app.services.preprocessing_service import clean_for_modeling
from app.utils.formatting import compact_number, percent


def _ensure_models() -> None:
    if st.session_state.training_results:
        return
    available = get_available_algorithms()
    algorithms = [name for name in ALGORITHMS if name in available]
    config = TrainingConfig(algorithms=algorithms, test_size=0.25, cv_folds=3)
    with st.spinner("Training baseline models for the dashboard..."):
        st.session_state.training_results = train_models(
            st.session_state.raw_df,
            st.session_state.selected_features,
            config,
        )
        st.session_state.active_model_name = best_model_name(st.session_state.training_results)


def render() -> None:
    """Render the analytics landing page."""
    page_header(
        "Dashboard",
        "Credit Risk Workflow Overview",
        "Portfolio health, model status, and interactive diagnostics in one workspace.",
    )
    _ensure_models()

    raw_df = st.session_state.raw_df
    clean_df = clean_for_modeling(raw_df)
    summary = summarize_dataset(raw_df)
    results = st.session_state.training_results
    best_name = best_model_name(results) or "Not trained"
    active_result = results.get(st.session_state.active_model_name or best_name)
    latest_prediction = st.session_state.last_prediction

    metric_values = [
        ("Loans", compact_number(summary.rows), "raw records"),
        ("Defaults", compact_number(summary.defaults), "observed target"),
        ("Default rate", percent(summary.default_rate), "portfolio imbalance"),
        ("Features", compact_number(len(clean_df.columns) - 1), "available fields"),
        ("Missing values", compact_number(summary.missing_values), "before cleaning"),
        ("Model status", "Ready" if results else "Not trained", f"{len(results)} model(s)"),
        ("Best algorithm", best_name, "ranked by ROC-AUC"),
        (
            "Latest prediction",
            percent(latest_prediction.probability) if latest_prediction else "None",
            latest_prediction.risk_category if latest_prediction else "no applicant scored",
        ),
    ]

    columns = st.columns(4)
    for index, (label, value, help_text) in enumerate(metric_values):
        columns[index % 4].metric(label, value, help=help_text)

    section_title("Portfolio Diagnostics", "Interactive charts summarize the current dataset and active model.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.loan_status_distribution(clean_df), use_container_width=True)
        if active_result:
            st.plotly_chart(charts.feature_importance(active_result.feature_importance), use_container_width=True)
    with right:
        st.plotly_chart(charts.class_imbalance(clean_df), use_container_width=True)
        st.plotly_chart(charts.model_comparison(metrics_frame(results)), use_container_width=True)

    st.plotly_chart(
        charts.correlation_heatmap(
            clean_df,
            columns=[col for col in st.session_state.selected_features if col in clean_df.columns and col != TARGET_COLUMN] + [TARGET_COLUMN],
            height=560,
        ),
        use_container_width=True,
    )
