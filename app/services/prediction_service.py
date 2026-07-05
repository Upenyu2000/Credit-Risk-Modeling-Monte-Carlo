# Author: Upenyu Hlangabeza
"""Prediction helpers for manual applicant scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import CATEGORICAL_OPTIONS, DEPLOYED_MODEL_FEATURES, MODEL_ARTIFACT_PATH
from app.models.schemas import PredictionResult, TrainingResult
from app.services.explainability_service import local_contributions
from app.services.preprocessing_service import add_engineered_features
from app.utils.formatting import risk_category


def default_applicant() -> dict[str, Any]:
    """Return default form values for the prediction page."""
    return {
        "age": 32,
        "gender": "Male",
        "marital_status": "Married",
        "employment_status": "Salaried",
        "income": 520000,
        "loan_amount": 1200000,
        "loan_tenure_months": 48,
        "total_loan_months": 96,
        "delinquent_months": 2,
        "total_dpd": 32,
        "credit_utilization_ratio": 42.0,
        "residence_type": "Owned",
        "loan_purpose": "Home",
        "loan_type": "Secured",
    }


def applicant_to_frame(applicant: dict[str, Any]) -> pd.DataFrame:
    """Convert form values to a single-row dataframe with engineered fields."""
    row = applicant.copy()
    row["avg_dpd_per_dm"] = round(row["total_dpd"] / row["delinquent_months"], 1) if row["delinquent_months"] else 0
    row["dmtlm"] = round((row["delinquent_months"] / max(row["total_loan_months"], 1)) * 100, 1)
    row["lti"] = round(row["loan_amount"] / max(row["income"], 1), 2)
    return add_engineered_features(pd.DataFrame([row]))


def _align_applicant_features(frame: pd.DataFrame, features: list[str], reference_df: pd.DataFrame) -> pd.DataFrame:
    """Fill active model features that are not part of the prediction form."""
    aligned = frame.copy()
    for feature in features:
        if feature in aligned.columns:
            continue
        if feature in reference_df.columns:
            series = reference_df[feature]
            if pd.api.types.is_numeric_dtype(series):
                aligned[feature] = series.median()
            else:
                mode = series.mode(dropna=True)
                aligned[feature] = mode.iloc[0] if not mode.empty else "Unknown"
        else:
            aligned[feature] = 0
    return aligned


def load_deployed_model_data() -> dict[str, Any] | None:
    """Load the existing project model artifact when dependencies are installed."""
    if not MODEL_ARTIFACT_PATH.exists():
        return None
    try:
        import joblib

        return joblib.load(MODEL_ARTIFACT_PATH)
    except Exception:
        return None


def predict_with_deployed_model(applicant: dict[str, Any]) -> float | None:
    """Score an applicant with the original serialized model artifact."""
    model_data = load_deployed_model_data()
    if not model_data:
        return None

    row = applicant_to_frame(applicant).iloc[0].to_dict()
    model_input = {
        "age": row["age"],
        "avg_dpd_per_dm": row["avg_dpd_per_dm"],
        "credit_utilization_ratio": row["credit_utilization_ratio"],
        "dmtlm": row["dmtlm"],
        "income": row["income"],
        "loan_amount": row["loan_amount"],
        "lti": row["lti"],
        "total_loan_months": row["total_loan_months"],
        "loan_tenure_months": row["loan_tenure_months"],
        "loan_purpose_Education": 1 if row["loan_purpose"] == "Education" else 0,
        "loan_purpose_Home": 1 if row["loan_purpose"] == "Home" else 0,
        "loan_purpose_Personal": 1 if row["loan_purpose"] == "Personal" else 0,
        "loan_type_Unsecured": 1 if row["loan_type"] == "Unsecured" else 0,
        "residence_type_Owned": 1 if row["residence_type"] == "Owned" else 0,
        "residence_type_Rented": 1 if row["residence_type"] == "Rented" else 0,
    }
    frame = pd.DataFrame([model_input]).reindex(columns=model_data.get("features", DEPLOYED_MODEL_FEATURES), fill_value=0)
    scale_cols = model_data.get("cols_to_scale", [])
    if scale_cols:
        frame[scale_cols] = model_data["scaler"].transform(frame[scale_cols])
    return float(model_data["model"].predict_proba(frame)[:, 1][0])


def predict_applicant(
    applicant: dict[str, Any],
    training_result: TrainingResult | None,
    reference_df: pd.DataFrame,
    prefer_deployed_model: bool = False,
) -> PredictionResult:
    """Predict default risk using either the original artifact or active session model."""
    frame = applicant_to_frame(applicant)
    model_name = training_result.name if training_result else "Deployed model"

    probability: float | None = None
    explanation = pd.DataFrame()

    if prefer_deployed_model:
        probability = predict_with_deployed_model(applicant)
        if probability is not None:
            model_name = "Original deployed XGBoost"

    if probability is None and training_result is not None:
        aligned = _align_applicant_features(frame, training_result.features, reference_df)
        probability = float(training_result.pipeline.predict_proba(aligned[training_result.features])[:, 1][0])
        explanation = local_contributions(training_result, reference_df, aligned.iloc[0])

    if probability is None:
        probability = heuristic_probability(applicant)
        model_name = "Heuristic fallback"

    credit_score = int(300 + (1 - probability) * 600)
    confidence = float(max(probability, 1 - probability))
    return PredictionResult(
        probability=probability,
        credit_score=credit_score,
        risk_category=risk_category(probability),
        confidence=confidence,
        applicant=applicant,
        explanation=explanation,
        model_name=model_name,
    )


def heuristic_probability(applicant: dict[str, Any]) -> float:
    """Fallback scoring if no trained or serialized model is available."""
    frame = applicant_to_frame(applicant).iloc[0]
    risk = (
        -3.0
        + 0.026 * frame["credit_utilization_ratio"]
        + 0.02 * frame["avg_dpd_per_dm"]
        + 0.034 * frame["dmtlm"]
        + 0.12 * frame["lti"]
        + 0.5 * (frame["loan_type"] == "Unsecured")
        + 0.2 * (frame["residence_type"] == "Rented")
    )
    return float(1 / (1 + np.exp(-risk)))


def option_list(column: str) -> list[str]:
    """Expose categorical options for prediction widgets."""
    return CATEGORICAL_OPTIONS[column]
