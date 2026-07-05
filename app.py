# Author: Upenyu Hlangabeza
"""Entry point for the interactive credit risk analytics platform."""

from __future__ import annotations

import streamlit as st

from app.components.layout import inject_theme, render_sidebar
from app.config import APP_TITLE
from app.pages import (
    dashboard,
    data_cleaning,
    dataset_explorer,
    eda,
    explainability,
    feature_engineering,
    model_evaluation,
    model_training,
    predictions,
    settings,
)
from app.utils.state import initialize_session_state

PAGE_RENDERERS = {
    "Dashboard": dashboard.render,
    "Dataset Explorer": dataset_explorer.render,
    "Data Cleaning": data_cleaning.render,
    "Exploratory Data Analysis": eda.render,
    "Feature Engineering": feature_engineering.render,
    "Model Training": model_training.render,
    "Model Evaluation": model_evaluation.render,
    "Predictions": predictions.render,
    "Explainability": explainability.render,
    "Settings": settings.render,
}


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="CR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    initialize_session_state()

    page_name = render_sidebar()
    render_page = PAGE_RENDERERS[page_name]
    render_page()


if __name__ == "__main__":
    main()
