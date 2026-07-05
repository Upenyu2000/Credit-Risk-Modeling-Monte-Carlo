# Author: Upenyu Hlangabeza
"""Streamlit session-state bootstrap helpers."""

from __future__ import annotations

import streamlit as st

from app.config import DEFAULT_SELECTED_FEATURES
from app.models.schemas import CleaningOptions
from app.services.data_service import generate_demo_credit_data


def initialize_session_state() -> None:
    """Create stable state defaults used across pages."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = generate_demo_credit_data()
        st.session_state.dataset_source = "Generated demo portfolio"

    if "cleaning_options" not in st.session_state:
        st.session_state.cleaning_options = CleaningOptions()

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = DEFAULT_SELECTED_FEATURES.copy()

    if "training_results" not in st.session_state:
        st.session_state.training_results = {}

    if "active_model_name" not in st.session_state:
        st.session_state.active_model_name = None

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "last_feature_signature" not in st.session_state:
        st.session_state.last_feature_signature = tuple(st.session_state.selected_features)
