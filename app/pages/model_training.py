# Author: Upenyu Hlangabeza
"""Model training page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.config import ALGORITHMS
from app.models.schemas import TrainingConfig
from app.services.model_service import best_model_name, get_available_algorithms, metrics_frame, train_single_model


def render() -> None:
    """Render model training controls."""
    page_header(
        "Model Training",
        "Train and Compare Algorithms",
        "Control algorithms, splits, cross-validation, and lightweight tuning without leaving the GUI.",
    )

    available = get_available_algorithms()
    missing = [name for name in ALGORITHMS if name not in available]
    if missing:
        st.warning(f"Unavailable optional algorithms: {', '.join(missing)}. Install the matching packages to enable them.")

    cols = st.columns([2, 1, 1, 1])
    algorithms = cols[0].multiselect("Algorithms", available, default=available)
    test_size = cols[1].slider("Test split", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    cv_folds = cols[2].selectbox("CV folds", [0, 3, 5], index=1)
    tune = cols[3].toggle("Tuning preset", value=False)

    if st.button("Train selected models", type="primary", disabled=not algorithms):
        progress = st.progress(0, text="Preparing training run...")
        results = {}
        for index, algorithm in enumerate(algorithms, start=1):
            progress.progress((index - 1) / len(algorithms), text=f"Training {algorithm}...")
            config = TrainingConfig(
                algorithms=[algorithm],
                test_size=test_size,
                cv_folds=cv_folds,
                tune_hyperparameters=tune,
            )
            results[algorithm] = train_single_model(st.session_state.raw_df, st.session_state.selected_features, config, algorithm)
        progress.progress(1.0, text="Training complete.")
        st.session_state.training_results = results
        st.session_state.active_model_name = best_model_name(results)
        st.session_state.last_feature_signature = tuple(st.session_state.selected_features)

    results = st.session_state.training_results
    metrics = metrics_frame(results)
    section_title("Training Metrics", "Scores are calculated on the holdout set; CV ROC-AUC is shown when enabled.")
    if metrics.empty:
        st.info("No trained models yet. Select algorithms and start training.")
        return

    display_metrics = metrics.copy()
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]:
        if column in display_metrics:
            display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
    st.dataframe(display_metrics, use_container_width=True)
    st.plotly_chart(charts.model_comparison(metrics), use_container_width=True)

    metric_cols = st.columns(len(results))
    for index, (name, result) in enumerate(results.items()):
        with metric_cols[index]:
            st.metric(name, f"{result.metrics['roc_auc']:.3f}", help="Holdout ROC-AUC")
