# Author: Upenyu Hlangabeza
"""Dataset loading, generation, and summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import pandas as pd

from app.config import CATEGORICAL_OPTIONS, TARGET_COLUMN


@dataclass(slots=True)
class DatasetSummary:
    """High-level portfolio summary."""

    rows: int
    columns: int
    defaults: int
    default_rate: float
    missing_values: int
    duplicate_rows: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_demo_credit_data(row_count: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible credit-risk portfolio when no raw CSV is available."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 68, row_count)
    income = rng.lognormal(mean=12.25, sigma=0.48, size=row_count).clip(120_000, 3_800_000).round(0)
    loan_tenure_months = rng.choice([12, 18, 24, 36, 48, 60, 84, 120, 180, 240], row_count, p=[0.06, 0.08, 0.13, 0.22, 0.16, 0.15, 0.08, 0.06, 0.04, 0.02])
    loan_multiplier = rng.gamma(shape=2.1, scale=1.05, size=row_count).clip(0.15, 8.5)
    loan_amount = (income * loan_multiplier).clip(45_000, 5_500_000).round(0)
    total_loan_months = (loan_tenure_months + rng.integers(0, 150, row_count)).clip(6, 360)

    gender = rng.choice(CATEGORICAL_OPTIONS["gender"], row_count, p=[0.43, 0.57])
    marital_status = rng.choice(CATEGORICAL_OPTIONS["marital_status"], row_count, p=[0.38, 0.62])
    employment_status = rng.choice(CATEGORICAL_OPTIONS["employment_status"], row_count, p=[0.73, 0.27])
    residence_type = rng.choice(CATEGORICAL_OPTIONS["residence_type"], row_count, p=[0.46, 0.34, 0.20])
    loan_purpose = rng.choice(CATEGORICAL_OPTIONS["loan_purpose"], row_count, p=[0.22, 0.26, 0.31, 0.21])
    loan_type = rng.choice(CATEGORICAL_OPTIONS["loan_type"], row_count, p=[0.67, 0.33])
    city = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai"], row_count)
    state = rng.choice(["MH", "DL", "KA", "TS", "TN"], row_count)

    credit_utilization_ratio = np.round(rng.beta(2.2, 4.1, row_count) * 100, 1)
    delinquency_pressure = _sigmoid((credit_utilization_ratio - 55) / 13 + (loan_type == "Unsecured") * 0.55)
    delinquent_months = rng.binomial(np.minimum(total_loan_months, 72), np.clip(delinquency_pressure * 0.085, 0.005, 0.22))
    total_dpd = np.where(delinquent_months > 0, rng.gamma(2.6, 18.0, row_count) * delinquent_months, 0).round(0)
    avg_dpd_per_dm = np.zeros(row_count, dtype=float)
    np.divide(total_dpd, delinquent_months, out=avg_dpd_per_dm, where=delinquent_months > 0)
    avg_dpd_per_dm = np.round(avg_dpd_per_dm, 1)
    dmtlm = np.round((delinquent_months / total_loan_months) * 100, 1)
    lti = np.round(loan_amount / np.maximum(income, 1), 2)

    processing_fee = np.round(loan_amount * rng.uniform(0.006, 0.032, row_count), 0)
    gst = np.round(processing_fee * 0.18, 0)
    net_disbursement = np.round(loan_amount - processing_fee - gst, 0)
    principal_outstanding = np.round(loan_amount * rng.uniform(0.04, 0.95, row_count), 0)
    bank_balance = np.round(income / 12 * rng.uniform(0.08, 2.8, row_count), 0)

    risk_score = (
        -3.45
        + 0.032 * credit_utilization_ratio
        + 0.020 * avg_dpd_per_dm
        + 0.035 * dmtlm
        + 0.145 * lti
        + 0.55 * (loan_type == "Unsecured")
        + 0.24 * (residence_type == "Rented")
        + 0.18 * (employment_status == "Self-employed")
        + 0.17 * (age < 30)
        - 0.00000022 * income
    )
    default_probability = _sigmoid(risk_score)
    default = rng.binomial(1, np.clip(default_probability, 0.01, 0.72))

    start = pd.Timestamp("2021-01-01")
    disbursal_date = start + pd.to_timedelta(rng.integers(0, 1250, row_count), unit="D")
    installment_start = disbursal_date + pd.to_timedelta(rng.integers(20, 65, row_count), unit="D")

    df = pd.DataFrame(
        {
            "cust_id": [f"CUST-{idx:06d}" for idx in range(1, row_count + 1)],
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "income": income,
            "loan_amount": loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "total_loan_months": total_loan_months,
            "delinquent_months": delinquent_months,
            "total_dpd": total_dpd,
            "avg_dpd_per_dm": avg_dpd_per_dm,
            "dmtlm": dmtlm,
            "credit_utilization_ratio": credit_utilization_ratio,
            "lti": lti,
            "residence_type": residence_type,
            "loan_purpose": loan_purpose,
            "loan_type": loan_type,
            "city": city,
            "state": state,
            "zipcode": rng.integers(100000, 999999, row_count).astype(str),
            "disbursal_date": disbursal_date,
            "installment_start_dt": installment_start,
            "processing_fee": processing_fee,
            "gst": gst,
            "net_disbursement": net_disbursement,
            "principal_outstanding": principal_outstanding,
            "bank_balance_at_application": bank_balance,
            TARGET_COLUMN: default,
        }
    )

    missing_columns = ["income", "credit_utilization_ratio", "employment_status", "bank_balance_at_application"]
    for column in missing_columns:
        mask = rng.random(row_count) < 0.018
        df.loc[mask, column] = np.nan

    df.loc[rng.choice(df.index, size=18, replace=False), "loan_purpose"] = "Personaal"
    outlier_rows = rng.choice(df.index, size=12, replace=False)
    df.loc[outlier_rows[:6], "income"] = df.loc[outlier_rows[:6], "income"] * 5
    df.loc[outlier_rows[6:], "loan_amount"] = df.loc[outlier_rows[6:], "loan_amount"] * 4

    duplicate_sample = df.sample(18, random_state=seed + 11)
    return pd.concat([df, duplicate_sample], ignore_index=True)


def read_uploaded_dataset(file: BinaryIO) -> pd.DataFrame:
    """Read an uploaded CSV file and normalize obvious date columns."""
    df = pd.read_csv(file)
    for column in ["disbursal_date", "installment_start_dt"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    """Return the dashboard summary for a dataset."""
    defaults = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df.columns else 0
    rate = float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df.columns and len(df) else 0.0
    return DatasetSummary(
        rows=len(df),
        columns=len(df.columns),
        defaults=defaults,
        default_rate=rate,
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )
