# Author: Upenyu Hlangabeza
"""Feature engineering page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
import pandas as pd

from app.config import ALGORITHMS
from app.models.schemas import TrainingConfig
from app.services.explainability_service import correlation_ranking
from app.services.model_service import best_model_name, get_available_algorithms, train_models
from app.services.preprocessing_service import candidate_model_features, clean_for_modeling


def _train_selected_features() -> None:
    available = get_available_algorithms()
    algorithms = [name for name in ALGORITHMS if name in available]
    config = TrainingConfig(algorithms=algorithms, test_size=0.25, cv_folds=3)
    with st.spinner("Retraining models with the selected feature set..."):
        st.session_state.training_results = train_models(st.session_state.raw_df, st.session_state.selected_features, config)
        st.session_state.active_model_name = best_model_name(st.session_state.training_results)
        st.session_state.last_feature_signature = tuple(st.session_state.selected_features)


def _auto_retrain_if_needed(enabled: bool) -> None:
    signature = tuple(st.session_state.selected_features)
    if not enabled or signature == st.session_state.last_feature_signature:
        return
    _train_selected_features()


def render() -> None:
    """Render feature controls and live retraining hooks."""
    page_header(
        "Feature Engineering",
        "Feature Selection and Signal Ranking",
        "Inspect engineered fields, enable or disable predictors, and retrain models from the UI.",
    )

    df = clean_for_modeling(st.session_state.raw_df)
    candidates = candidate_model_features(df)
    selected = st.multiselect("Selected model features", candidates, default=[f for f in st.session_state.selected_features if f in candidates])
    if selected:
        st.session_state.selected_features = selected

    auto_retrain = st.toggle("Retrain immediately when feature selection changes", value=False)
    _auto_retrain_if_needed(auto_retrain)

    if st.button("Retrain with selected features", type="primary"):
        _train_selected_features()

    section_title("Engineered Features", "The notebook workflow creates ratios that encode affordability and delinquency behavior.")
    cols = st.columns(3)
    cols[0].metric("LTI", "loan_amount / income", help="Loan-to-income ratio captures affordability pressure.")
    cols[1].metric("DMTLM", "delinquent months %", help="Share of loan months with delinquency.")
    cols[2].metric("Avg DPD", "total DPD / delinquent months", help="Average delinquency severity.")

    left, right = st.columns(2)
    with left:
        ranking = correlation_ranking(df)
        st.dataframe(ranking.head(16), use_container_width=True, height=420)
    with right:
        active = st.session_state.training_results.get(st.session_state.active_model_name)
        importance = active.feature_importance if active else pd.DataFrame(columns=["feature", "importance"])
        st.plotly_chart(charts.feature_importance(importance), use_container_width=True)

    section_title("Selected Feature Preview", f"{len(st.session_state.selected_features)} active features")
    st.dataframe(df[st.session_state.selected_features + ["default"]].head(18), use_container_width=True)
