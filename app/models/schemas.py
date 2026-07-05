# Author: Upenyu Hlangabeza
"""Typed data structures shared across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class CleaningOptions:
    """Interactive preprocessing switches used by the cleaning page."""

    remove_duplicates: bool = True
    handle_missing: bool = True
    encode_categoricals: bool = False
    scale_numerics: bool = False
    remove_outliers: bool = False


@dataclass(slots=True)
class CleaningStep:
    """Before-and-after details for one preprocessing operation."""

    name: str
    before_rows: int
    after_rows: int
    before_missing: int
    after_missing: int
    details: str


@dataclass(slots=True)
class TrainingConfig:
    """Model training controls selected in the UI."""

    algorithms: list[str]
    test_size: float = 0.25
    cv_folds: int = 3
    tune_hyperparameters: bool = False
    random_state: int = 42


@dataclass(slots=True)
class TrainingResult:
    """Trained model, metrics, and evaluation artifacts."""

    name: str
    pipeline: Any
    features: list[str]
    metrics: dict[str, float]
    classification_report: pd.DataFrame
    confusion_matrix: Any
    y_test: Any
    y_pred: Any
    y_proba: Any
    feature_importance: pd.DataFrame
    cv_score: float | None = None


@dataclass(slots=True)
class PredictionResult:
    """Prediction output displayed by the prediction and explainability pages."""

    probability: float
    credit_score: int
    risk_category: str
    confidence: float
    applicant: dict[str, Any]
    explanation: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_name: str = "Session model"
