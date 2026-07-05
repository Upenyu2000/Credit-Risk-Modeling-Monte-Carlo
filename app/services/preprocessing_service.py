# Author: Upenyu Hlangabeza
"""Preprocessing and feature-engineering services."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import DATE_COLUMNS, ID_COLUMNS, RAW_FEATURES, TARGET_COLUMN
from app.models.schemas import CleaningOptions, CleaningStep


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure core engineered risk features are present."""
    result = df.copy()
    if {"loan_amount", "income"}.issubset(result.columns):
        result["lti"] = np.round(result["loan_amount"] / result["income"].replace(0, np.nan), 2).fillna(0)
    if {"delinquent_months", "total_loan_months"}.issubset(result.columns):
        result["dmtlm"] = np.round((result["delinquent_months"] / result["total_loan_months"].replace(0, np.nan)) * 100, 1).fillna(0)
    if {"total_dpd", "delinquent_months"}.issubset(result.columns):
        delinquent = result["delinquent_months"].to_numpy(dtype=float)
        total_dpd = result["total_dpd"].to_numpy(dtype=float)
        avg_dpd = np.zeros(len(result), dtype=float)
        np.divide(total_dpd, delinquent, out=avg_dpd, where=delinquent > 0)
        result["avg_dpd_per_dm"] = np.round(avg_dpd, 1)
    return result


def fix_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known categorical inconsistencies from the notebooks."""
    result = df.copy()
    if "loan_purpose" in result.columns:
        result["loan_purpose"] = result["loan_purpose"].replace({"Personaal": "Personal", "personal": "Personal"})
    return result


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns excluding target-like fields."""
    return [col for col in df.select_dtypes(include=np.number).columns if col != TARGET_COLUMN]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for exploration and modeling."""
    excluded = ID_COLUMNS | DATE_COLUMNS
    return [col for col in df.select_dtypes(include=["object", "category"]).columns if col not in excluded]


def candidate_model_features(df: pd.DataFrame) -> list[str]:
    """Return modelable feature columns in a stable order."""
    preferred = [feature for feature in RAW_FEATURES if feature in df.columns]
    extras = [
        col
        for col in df.columns
        if col not in preferred and col not in ID_COLUMNS and col not in DATE_COLUMNS and col != TARGET_COLUMN
    ]
    return preferred + extras


def apply_cleaning(df: pd.DataFrame, options: CleaningOptions) -> tuple[pd.DataFrame, list[CleaningStep]]:
    """Apply interactive preprocessing options and describe every step."""
    result = add_engineered_features(fix_categorical_values(df))
    steps: list[CleaningStep] = []

    def record(name: str, before: pd.DataFrame, after: pd.DataFrame, details: str) -> None:
        steps.append(
            CleaningStep(
                name=name,
                before_rows=len(before),
                after_rows=len(after),
                before_missing=int(before.isna().sum().sum()),
                after_missing=int(after.isna().sum().sum()),
                details=details,
            )
        )

    if options.remove_duplicates:
        before = result.copy()
        result = result.drop_duplicates().reset_index(drop=True)
        record("Remove duplicates", before, result, "Drops exact duplicate borrower records.")

    if options.handle_missing:
        before = result.copy()
        for column in result.columns:
            if result[column].isna().any():
                if pd.api.types.is_numeric_dtype(result[column]):
                    result[column] = result[column].fillna(result[column].median())
                else:
                    mode = result[column].mode(dropna=True)
                    result[column] = result[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        record("Handle missing values", before, result, "Uses median imputation for numerics and mode imputation for categoricals.")

    if options.remove_outliers:
        before = result.copy()
        for column in get_numeric_columns(result):
            q1, q3 = result[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            result = result[(result[column] >= lower) & (result[column] <= upper)]
        result = result.reset_index(drop=True)
        record("Remove outliers", before, result, "Filters numeric values outside 1.5x IQR bounds.")

    if options.encode_categoricals:
        before = result.copy()
        categorical = get_categorical_columns(result)
        result = pd.get_dummies(result, columns=categorical, drop_first=True, dtype=int)
        record("Encode categoricals", before, result, "One-hot encodes modelable categorical fields.")

    if options.scale_numerics:
        before = result.copy()
        numeric = get_numeric_columns(result)
        for column in numeric:
            std = result[column].std()
            if std and not np.isnan(std):
                result[column] = (result[column] - result[column].mean()) / std
        record("Scale numeric variables", before, result, "Applies z-score scaling to numeric feature columns.")

    return result, steps


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the stable preprocessing baseline used by model training."""
    options = CleaningOptions(
        remove_duplicates=True,
        handle_missing=True,
        encode_categoricals=False,
        scale_numerics=False,
        remove_outliers=False,
    )
    result, _ = apply_cleaning(df, options)
    return result


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact descriptive statistics table."""
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.round(2)


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return data types, missing counts, and uniqueness."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[column].dtype) for column in df.columns],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
            "unique": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )
