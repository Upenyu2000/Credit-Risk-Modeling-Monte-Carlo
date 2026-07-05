# Author: Upenyu Hlangabeza
"""Model training, scoring, and evaluation services."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import TARGET_COLUMN
from app.models.schemas import TrainingConfig, TrainingResult
from app.services.preprocessing_service import candidate_model_features, clean_for_modeling

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


def get_available_algorithms() -> list[str]:
    """Return algorithms available in the current environment."""
    algorithms = ["Logistic Regression", "Random Forest"]
    if XGBClassifier is not None:
        algorithms.append("XGBoost")
    return algorithms


def make_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible with modern scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old scikit-learn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_estimator(name: str, tune_hyperparameters: bool, random_state: int):
    """Return the estimator associated with a UI algorithm name."""
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            C=0.7 if tune_hyperparameters else 1.0,
            random_state=random_state,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=180 if tune_hyperparameters else 110,
            max_depth=9 if tune_hyperparameters else None,
            min_samples_leaf=3 if tune_hyperparameters else 1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "XGBoost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=170 if tune_hyperparameters else 95,
            max_depth=4 if tune_hyperparameters else 3,
            learning_rate=0.055 if tune_hyperparameters else 0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    raise ValueError(f"Unsupported or unavailable algorithm: {name}")


def build_pipeline(df: pd.DataFrame, features: list[str], estimator) -> Pipeline:
    """Build preprocessing and estimator pipeline for mixed feature types."""
    selected = df[features]
    numeric = selected.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in selected.columns if column not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", make_encoder(), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _predict_probability(pipeline: Pipeline, x_values: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_values)[:, 1]
    scores = pipeline.decision_function(x_values)
    return 1 / (1 + np.exp(-scores))


def _feature_names(pipeline: Pipeline, features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return features


def _feature_importance(pipeline: Pipeline, features: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline, features)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        values = np.zeros(len(names), dtype=float)
    if len(values) != len(names):
        names = names[: len(values)]
    total = values.sum()
    normalized = values / total if total else values
    return pd.DataFrame({"feature": names, "importance": normalized}).sort_values("importance", ascending=False)


def train_single_model(df: pd.DataFrame, features: list[str], config: TrainingConfig, algorithm: str) -> TrainingResult:
    """Train and evaluate one model."""
    clean_df = clean_for_modeling(df)
    usable_features = [feature for feature in features if feature in clean_df.columns]
    if not usable_features:
        usable_features = candidate_model_features(clean_df)

    model_df = clean_df[usable_features + [TARGET_COLUMN]].dropna(subset=[TARGET_COLUMN])
    x_values = model_df[usable_features]
    y_values = model_df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=config.test_size,
        stratify=y_values,
        random_state=config.random_state,
    )

    estimator = build_estimator(algorithm, config.tune_hyperparameters, config.random_state)
    pipeline = build_pipeline(model_df, usable_features, estimator)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = _predict_probability(pipeline, x_test)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    cv_score = None
    if config.cv_folds >= 2:
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_score = float(cross_val_score(pipeline, x_values, y_values, cv=cv, scoring="roc_auc").mean())

    return TrainingResult(
        name=algorithm,
        pipeline=pipeline,
        features=usable_features,
        metrics={
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc),
        },
        classification_report=pd.DataFrame(report_dict).T.round(3),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        y_test=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
        feature_importance=_feature_importance(pipeline, usable_features),
        cv_score=cv_score,
    )


def train_models(df: pd.DataFrame, features: list[str], config: TrainingConfig) -> dict[str, TrainingResult]:
    """Train all requested models."""
    results: dict[str, TrainingResult] = {}
    for algorithm in config.algorithms:
        results[algorithm] = train_single_model(df, features, config, algorithm)
    return results


def metrics_frame(results: dict[str, TrainingResult]) -> pd.DataFrame:
    """Convert model metrics into a comparison table."""
    rows = []
    for name, result in results.items():
        row = {"model": name, **result.metrics}
        row["cv_roc_auc"] = result.cv_score
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False) if rows else pd.DataFrame()


def best_model_name(results: dict[str, TrainingResult]) -> str | None:
    """Return the highest ROC-AUC model name."""
    if not results:
        return None
    return max(results.items(), key=lambda item: item[1].metrics.get("roc_auc", 0))[0]
