# Author: Upenyu Hlangabeza
"""Global and local explainability utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.schemas import TrainingResult
from app.services.preprocessing_service import clean_for_modeling


def global_importance(result: TrainingResult | None) -> pd.DataFrame:
    """Return model feature importance in a display-ready dataframe."""
    if result is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return result.feature_importance.copy()


def local_contributions(result: TrainingResult, reference_df: pd.DataFrame, applicant_row: pd.Series, top_n: int = 12) -> pd.DataFrame:
    """Estimate local feature contribution by replacing one feature at a time with a baseline value."""
    clean_reference = clean_for_modeling(reference_df)
    selected = [feature for feature in result.features if feature in applicant_row.index and feature in clean_reference.columns]
    if not selected:
        return pd.DataFrame(columns=["feature", "impact", "baseline", "value"])

    row = pd.DataFrame([applicant_row[selected].to_dict()])
    base_probability = float(result.pipeline.predict_proba(row[selected])[:, 1][0])
    records = []

    for feature in selected:
        comparison = row.copy()
        comparison[feature] = comparison[feature].astype(object)
        series = clean_reference[feature]
        if pd.api.types.is_numeric_dtype(series):
            baseline = float(series.median())
        else:
            mode = series.mode(dropna=True)
            baseline = mode.iloc[0] if not mode.empty else series.dropna().iloc[0]
        comparison.loc[comparison.index[0], feature] = baseline
        shifted_probability = float(result.pipeline.predict_proba(comparison[selected])[:, 1][0])
        records.append(
            {
                "feature": feature,
                "impact": base_probability - shifted_probability,
                "baseline": baseline,
                "value": applicant_row[feature],
            }
        )

    explanation = pd.DataFrame(records)
    explanation["abs_impact"] = np.abs(explanation["impact"])
    return explanation.sort_values("abs_impact", ascending=False).head(top_n).drop(columns=["abs_impact"])


def correlation_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric features by absolute correlation with default."""
    clean_df = clean_for_modeling(df)
    numeric = clean_df.select_dtypes(include=np.number)
    if "default" not in numeric.columns:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])
    corr = numeric.corr(numeric_only=True)["default"].drop("default").dropna()
    ranking = corr.reset_index()
    ranking.columns = ["feature", "correlation"]
    ranking["abs_correlation"] = ranking["correlation"].abs()
    return ranking.sort_values("abs_correlation", ascending=False)
