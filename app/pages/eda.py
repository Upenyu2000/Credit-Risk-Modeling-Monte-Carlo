# Author: Upenyu Hlangabeza
"""Exploratory data analysis page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.config import TARGET_COLUMN
from app.services.preprocessing_service import clean_for_modeling, get_categorical_columns, get_numeric_columns


def render() -> None:
    """Render interactive EDA visualizations."""
    page_header(
        "Exploratory Data Analysis",
        "Interactive Visual Diagnostics",
        "Choose variables and chart types to explore distributions, relationships, and default patterns.",
    )

    df = clean_for_modeling(st.session_state.raw_df)
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)

    chart_type = st.selectbox(
        "Visualization",
        ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Correlation Matrix", "Pair Plot", "Loan Status Comparison"],
    )

    if chart_type == "Histogram":
        feature = st.selectbox("Feature", numeric, index=numeric.index("income") if "income" in numeric else 0)
        st.plotly_chart(charts.histogram(df, feature), use_container_width=True)
    elif chart_type == "Box Plot":
        feature = st.selectbox("Feature", numeric, index=numeric.index("loan_amount") if "loan_amount" in numeric else 0)
        st.plotly_chart(charts.box_plot(df, feature), use_container_width=True)
    elif chart_type == "Violin Plot":
        feature = st.selectbox("Feature", numeric, index=numeric.index("credit_utilization_ratio") if "credit_utilization_ratio" in numeric else 0)
        st.plotly_chart(charts.violin_plot(df, feature), use_container_width=True)
    elif chart_type == "Scatter Plot":
        default_x = numeric.index("income") if "income" in numeric else 0
        default_y = numeric.index("loan_amount") if "loan_amount" in numeric else min(1, len(numeric) - 1)
        x_axis = st.selectbox("X axis", numeric, index=default_x)
        y_axis = st.selectbox("Y axis", numeric, index=default_y)
        color = st.selectbox("Color", [TARGET_COLUMN] + categorical, index=0)
        st.plotly_chart(charts.scatter_plot(df, x_axis, y_axis, color=color), use_container_width=True)
    elif chart_type == "Correlation Matrix":
        selected = st.multiselect("Features", numeric + [TARGET_COLUMN], default=[col for col in ["income", "loan_amount", "credit_utilization_ratio", "avg_dpd_per_dm", "dmtlm", "lti", TARGET_COLUMN] if col in df.columns])
        st.plotly_chart(charts.correlation_heatmap(df, selected, height=620), use_container_width=True)
    elif chart_type == "Pair Plot":
        default_features = [col for col in ["income", "loan_amount", "credit_utilization_ratio", "lti"] if col in numeric]
        selected = st.multiselect("Up to four numeric features", numeric, default=default_features[:4], max_selections=4)
        if len(selected) >= 2:
            st.plotly_chart(charts.scatter_matrix(df.sample(min(len(df), 800), random_state=7), selected), use_container_width=True)
        else:
            st.info("Select at least two numeric features.")
    else:
        category = st.selectbox("Category", categorical, index=categorical.index("loan_purpose") if "loan_purpose" in categorical else 0)
        st.plotly_chart(charts.categorical_default_rate(df, category), use_container_width=True)

    section_title("Feature Distributions", "Quick profile of key numeric fields.")
    cols = st.columns(3)
    for index, feature in enumerate([col for col in ["income", "loan_amount", "credit_utilization_ratio"] if col in df.columns]):
        with cols[index]:
            st.plotly_chart(charts.histogram(df, feature), use_container_width=True)
