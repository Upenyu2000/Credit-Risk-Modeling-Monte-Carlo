# Author: Upenyu Hlangabeza
"""Dataset explorer page."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app.components.layout import page_header, section_title
from app.services.preprocessing_service import apply_cleaning, descriptive_stats, dtype_summary, get_categorical_columns


def _filter_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    mask = df.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False).any(), axis=1)
    return df[mask]


def _style_missing(df: pd.DataFrame):
    return df.style.map(lambda value: "background-color: rgba(214, 103, 103, 0.28)" if pd.isna(value) else "")


def render() -> None:
    """Render interactive dataset browsing tools."""
    page_header(
        "Dataset Explorer",
        "Browse, Search, Filter, and Inspect",
        "A table-first workspace for understanding the credit portfolio before modeling.",
    )

    source = st.radio("Dataset view", ["Raw dataset", "Current cleaned view"], horizontal=True)
    if source == "Current cleaned view":
        df, _ = apply_cleaning(st.session_state.raw_df, st.session_state.cleaning_options)
    else:
        df = st.session_state.raw_df.copy()

    controls = st.columns([2, 1, 1, 1])
    search = controls[0].text_input("Search rows", placeholder="Search across all visible columns")
    sort_column = controls[1].selectbox("Sort by", df.columns.tolist(), index=0)
    sort_order = controls[2].selectbox("Order", ["Ascending", "Descending"])
    page_size = controls[3].selectbox("Rows per page", [10, 25, 50, 100], index=1)

    filtered = _filter_search(df, search)
    categorical = get_categorical_columns(filtered)
    if categorical:
        filter_column = st.selectbox("Filter categorical column", ["No filter"] + categorical)
        if filter_column != "No filter":
            values = sorted(filtered[filter_column].dropna().astype(str).unique().tolist())
            selected_values = st.multiselect("Allowed values", values, default=values[: min(len(values), 4)])
            if selected_values:
                filtered = filtered[filtered[filter_column].astype(str).isin(selected_values)]

    filtered = filtered.sort_values(sort_column, ascending=sort_order == "Ascending", kind="mergesort")
    total_pages = max(1, math.ceil(len(filtered) / page_size))
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page_number - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    section_title("Data Table", f"Showing {len(page_df):,} of {len(filtered):,} matching rows.")
    highlight_missing = st.toggle("Highlight missing values", value=True)
    st.dataframe(_style_missing(page_df) if highlight_missing else page_df, use_container_width=True, height=470)

    stats_tab, types_tab, missing_tab = st.tabs(["Descriptive Statistics", "Data Types", "Missing Values"])
    with stats_tab:
        st.dataframe(descriptive_stats(filtered), use_container_width=True)
    with types_tab:
        st.dataframe(dtype_summary(filtered), use_container_width=True)
    with missing_tab:
        missing = filtered.isna().sum().reset_index()
        missing.columns = ["column", "missing_values"]
        missing["missing_rate"] = (missing["missing_values"] / max(len(filtered), 1)).round(4)
        st.dataframe(missing.sort_values("missing_values", ascending=False), use_container_width=True)
