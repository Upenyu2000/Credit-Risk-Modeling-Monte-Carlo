# Author: Upenyu Hlangabeza
"""Plotly chart factories used across workflow pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from app.config import TARGET_COLUMN
from app.utils.formatting import titleize

PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#4f8cff"
PANEL = "#171a21"
GRID = "#2a303b"
GOOD = "#5dbb7a"
WARN = "#d7a74f"
RISK = "#d66767"


def _layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": "#e5e7eb", "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 32, "r": 20, "t": 48, "b": 34},
        height=height,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def loan_status_distribution(df: pd.DataFrame) -> go.Figure:
    labels = df[TARGET_COLUMN].map({0: "Non-default", 1: "Default"})
    counts = labels.value_counts().reset_index()
    counts.columns = ["Status", "Loans"]
    fig = px.bar(counts, x="Status", y="Loans", color="Status", color_discrete_map={"Non-default": GOOD, "Default": RISK})
    fig.update_traces(marker_line_width=0)
    return _layout(fig, 320)


def class_imbalance(df: pd.DataFrame) -> go.Figure:
    rate = df[TARGET_COLUMN].mean()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rate * 100,
            number={"suffix": "%", "font": {"size": 30}},
            title={"text": "Default Rate"},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 10], "color": "rgba(93,187,122,0.25)"},
                    {"range": [10, 25], "color": "rgba(215,167,79,0.25)"},
                    {"range": [25, 50], "color": "rgba(214,103,103,0.25)"},
                ],
            },
        )
    )
    return _layout(fig, 320)


def correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None, height: int = 480) -> go.Figure:
    numeric = df.select_dtypes(include=np.number)
    if columns:
        numeric = numeric[[col for col in columns if col in numeric.columns]]
    corr = numeric.corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    return _layout(fig, height)


def model_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return empty_chart("Train a model to compare performance.")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available = [col for col in metric_cols if col in metrics_df.columns]
    long_df = metrics_df.melt(id_vars="model", value_vars=available, var_name="Metric", value_name="Score")
    fig = px.bar(long_df, x="model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
    return _layout(fig, 390)


def feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    if importance_df.empty:
        return empty_chart("No feature importance is available yet.")
    top = importance_df.sort_values("importance", ascending=False).head(14).sort_values("importance")
    fig = px.bar(top, x="importance", y="feature", orientation="h", title=title, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0)
    return _layout(fig, 430)


def histogram(df: pd.DataFrame, feature: str, color: str | None = TARGET_COLUMN) -> go.Figure:
    color_column = color if color in df.columns else None
    fig = px.histogram(df, x=feature, color=color_column, nbins=36, marginal="box", barmode="overlay")
    return _layout(fig, 410)


def box_plot(df: pd.DataFrame, feature: str, group: str = TARGET_COLUMN) -> go.Figure:
    fig = px.box(df, x=group, y=feature, color=group, points="outliers")
    return _layout(fig, 410)


def violin_plot(df: pd.DataFrame, feature: str, group: str = TARGET_COLUMN) -> go.Figure:
    fig = px.violin(df, x=group, y=feature, color=group, box=True, points=False)
    return _layout(fig, 410)


def scatter_plot(df: pd.DataFrame, x_axis: str, y_axis: str, color: str = TARGET_COLUMN) -> go.Figure:
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color if color in df.columns else None, opacity=0.72)
    return _layout(fig, 430)


def scatter_matrix(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    fig = px.scatter_matrix(df, dimensions=columns, color=TARGET_COLUMN if TARGET_COLUMN in df.columns else None)
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker={"size": 4, "opacity": 0.55})
    return _layout(fig, 620)


def categorical_default_rate(df: pd.DataFrame, category: str) -> go.Figure:
    grouped = (
        df.groupby(category, dropna=False)[TARGET_COLUMN]
        .agg(default_rate="mean", loans="size")
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    fig = px.bar(grouped, x=category, y="default_rate", text="loans", color_discrete_sequence=[ACCENT])
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, 390)


def confusion_matrix(cm: np.ndarray, labels: list[str] | None = None) -> go.Figure:
    labels = labels or ["Non-default", "Default"]
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues")
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return _layout(fig, 380)


def roc_curve_chart(y_test: np.ndarray, y_proba: np.ndarray, name: str) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC {score:.3f}", line={"color": ACCENT, "width": 3}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line={"color": "#6b7280", "dash": "dash"}))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _layout(fig, 390)


def precision_recall_chart(y_test: np.ndarray, y_proba: np.ndarray, name: str) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name, line={"color": ACCENT, "width": 3}))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _layout(fig, 390)


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Default Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 8], "color": "rgba(93,187,122,0.25)"},
                    {"range": [8, 18], "color": "rgba(79,140,255,0.18)"},
                    {"range": [18, 35], "color": "rgba(215,167,79,0.25)"},
                    {"range": [35, 100], "color": "rgba(214,103,103,0.25)"},
                ],
            },
        )
    )
    return _layout(fig, 330)


def contribution_chart(explanation: pd.DataFrame) -> go.Figure:
    if explanation.empty:
        return empty_chart("Run a prediction to view local feature contributions.")
    plot_df = explanation.sort_values("impact")
    colors = [RISK if value > 0 else GOOD for value in plot_df["impact"]]
    fig = go.Figure(go.Bar(x=plot_df["impact"], y=plot_df["feature"].map(titleize), orientation="h", marker={"color": colors}))
    fig.update_layout(xaxis_title="Change in default probability", yaxis_title="")
    return _layout(fig, 420)


def missing_values_chart(before: pd.DataFrame, after: pd.DataFrame) -> go.Figure:
    summary = pd.DataFrame(
        {
            "Feature": before.columns,
            "Before": before.isna().sum().values,
            "After": after.reindex(columns=before.columns).isna().sum().values,
        }
    )
    summary = summary[(summary["Before"] > 0) | (summary["After"] > 0)].head(18)
    if summary.empty:
        return empty_chart("No missing values detected.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["Feature"], y=summary["Before"], name="Before", marker_color=WARN))
    fig.add_trace(go.Bar(x=summary["Feature"], y=summary["After"], name="After", marker_color=ACCENT))
    fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Missing values")
    return _layout(fig, 380)


def empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 15, "color": "#9aa4b2"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, 320)
