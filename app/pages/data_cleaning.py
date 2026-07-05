# Author: Upenyu Hlangabeza
"""Data cleaning page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, render_step, section_title
from app.models.schemas import CleaningOptions
from app.services.data_service import summarize_dataset
from app.services.preprocessing_service import apply_cleaning
from app.utils.formatting import compact_number


def render() -> None:
    """Render interactive cleaning controls and before-after comparisons."""
    page_header(
        "Data Cleaning",
        "Preprocessing Control Room",
        "Switch cleaning steps on or off and inspect exactly how the dataset changes.",
    )

    current: CleaningOptions = st.session_state.cleaning_options
    controls = st.columns(5)
    options = CleaningOptions(
        remove_duplicates=controls[0].toggle("Remove duplicates", value=current.remove_duplicates),
        handle_missing=controls[1].toggle("Handle missing", value=current.handle_missing),
        encode_categoricals=controls[2].toggle("Encode categoricals", value=current.encode_categoricals),
        scale_numerics=controls[3].toggle("Scale numerics", value=current.scale_numerics),
        remove_outliers=controls[4].toggle("Remove outliers", value=current.remove_outliers),
    )
    st.session_state.cleaning_options = options

    before = st.session_state.raw_df
    after, steps = apply_cleaning(before, options)
    before_summary = summarize_dataset(before)
    after_summary = summarize_dataset(after) if "default" in after.columns else before_summary

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows before", compact_number(before_summary.rows))
    metric_cols[1].metric("Rows after", compact_number(len(after)))
    metric_cols[2].metric("Missing before", compact_number(before_summary.missing_values))
    metric_cols[3].metric("Missing after", compact_number(int(after.isna().sum().sum())))

    section_title("Preprocessing Steps", "Each active operation is displayed with its row and missing-value impact.")
    if steps:
        for step in steps:
            render_step(step.name, step.details, step.before_rows, step.after_rows)
    else:
        st.info("No cleaning steps are enabled.")

    left, right = st.columns(2)
    with left:
        section_title("Before", "Raw dataset preview.")
        st.dataframe(before.head(12), use_container_width=True, height=350)
    with right:
        section_title("After", "Dataset preview after selected preprocessing.")
        st.dataframe(after.head(12), use_container_width=True, height=350)

    st.plotly_chart(charts.missing_values_chart(before, after), use_container_width=True)

    with st.expander("Cleaning Notes", expanded=False):
        st.write(
            "The original notebooks remove duplicate records, impute small amounts of missing data, "
            "normalize inconsistent categorical values, engineer LTI/DMTLM/DPD features, and optionally "
            "filter numeric outliers. This page makes those decisions visible and reversible."
        )
