# Author: Upenyu Hlangabeza
"""Application configuration and page registry."""

from __future__ import annotations

from pathlib import Path
APP_TITLE = "Credit Risk Analysis Studio"
APP_SUBTITLE = "Interactive financial machine learning workflow"

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "app" / "assets"
THEME_PATH = ASSETS_DIR / "theme.css"
MODEL_ARTIFACT_PATH = ROOT_DIR / "project-root" / "model" / "model_data.pkl"

TARGET_COLUMN = "default"
ID_COLUMNS = {"cust_id", "zipcode"}
DATE_COLUMNS = {"disbursal_date", "installment_start_dt"}

RAW_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "loan_tenure_months",
    "total_loan_months",
    "credit_utilization_ratio",
    "delinquent_months",
    "total_dpd",
    "avg_dpd_per_dm",
    "dmtlm",
    "lti",
    "gender",
    "marital_status",
    "employment_status",
    "residence_type",
    "loan_purpose",
    "loan_type",
]

DEPLOYED_MODEL_FEATURES = [
    "age",
    "avg_dpd_per_dm",
    "credit_utilization_ratio",
    "dmtlm",
    "income",
    "loan_amount",
    "lti",
    "total_loan_months",
    "loan_tenure_months",
    "loan_purpose_Education",
    "loan_purpose_Home",
    "loan_purpose_Personal",
    "loan_type_Unsecured",
    "residence_type_Owned",
    "residence_type_Rented",
]

DEFAULT_SELECTED_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "loan_tenure_months",
    "total_loan_months",
    "credit_utilization_ratio",
    "avg_dpd_per_dm",
    "dmtlm",
    "lti",
    "residence_type",
    "loan_purpose",
    "loan_type",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "marital_status": ["Single", "Married"],
    "employment_status": ["Salaried", "Self-employed"],
    "residence_type": ["Owned", "Rented", "Mortgage"],
    "loan_purpose": ["Education", "Home", "Auto", "Personal"],
    "loan_type": ["Secured", "Unsecured"],
}

ALGORITHMS = ["Logistic Regression", "Random Forest", "XGBoost"]
FALLBACK_ALGORITHMS = ["Logistic Regression", "Random Forest"]

PAGE_NAMES = [
    "Dashboard",
    "Dataset Explorer",
    "Data Cleaning",
    "Exploratory Data Analysis",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Predictions",
    "Explainability",
    "Settings",
]
