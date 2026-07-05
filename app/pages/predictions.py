# Author: Upenyu Hlangabeza
"""Prediction page."""

from __future__ import annotations

import streamlit as st

from app.components import charts
from app.components.layout import page_header, section_title
from app.services.prediction_service import default_applicant, option_list, predict_applicant
from app.services.preprocessing_service import clean_for_modeling
from app.utils.formatting import percent


def _prediction_form() -> dict:
    defaults = default_applicant()
    with st.form("prediction_form"):
        st.caption("Applicant profile")
        row1 = st.columns(3)
        age = row1[0].number_input("Age", 18, 100, defaults["age"])
        income = row1[1].number_input("Annual income", 0, 5_000_000, defaults["income"], step=25_000)
        loan_amount = row1[2].number_input("Loan amount", 0, 8_000_000, defaults["loan_amount"], step=25_000)

        row2 = st.columns(4)
        loan_tenure = row2[0].slider("Loan tenure months", 6, 240, defaults["loan_tenure_months"], step=6)
        total_months = row2[1].number_input("Total loan months", 0, 360, defaults["total_loan_months"])
        delinquent_months = row2[2].number_input("Delinquent months", 0, 120, defaults["delinquent_months"])
        total_dpd = row2[3].number_input("Total DPD", 0, 5000, defaults["total_dpd"])

        row3 = st.columns(4)
        utilization = row3[0].slider("Credit utilization %", 0.0, 100.0, defaults["credit_utilization_ratio"], step=1.0)
        loan_purpose = row3[1].selectbox("Loan purpose", option_list("loan_purpose"), index=option_list("loan_purpose").index(defaults["loan_purpose"]))
        loan_type = row3[2].selectbox("Loan type", option_list("loan_type"), index=option_list("loan_type").index(defaults["loan_type"]))
        residence = row3[3].selectbox("Residence", option_list("residence_type"), index=option_list("residence_type").index(defaults["residence_type"]))

        row4 = st.columns(3)
        gender = row4[0].selectbox("Gender", option_list("gender"), index=option_list("gender").index(defaults["gender"]))
        marital = row4[1].selectbox("Marital status", option_list("marital_status"), index=option_list("marital_status").index(defaults["marital_status"]))
        employment = row4[2].selectbox("Employment", option_list("employment_status"), index=option_list("employment_status").index(defaults["employment_status"]))

        prefer_deployed = st.toggle("Prefer original deployed model artifact when available", value=False)
        submitted = st.form_submit_button("Predict", type="primary")

    applicant = {
        "age": age,
        "gender": gender,
        "marital_status": marital,
        "employment_status": employment,
        "income": income,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure,
        "total_loan_months": total_months,
        "delinquent_months": delinquent_months,
        "total_dpd": total_dpd,
        "credit_utilization_ratio": utilization,
        "residence_type": residence,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "_submitted": submitted,
        "_prefer_deployed": prefer_deployed,
    }
    return applicant


def render() -> None:
    """Render manual applicant scoring workflow."""
    page_header(
        "Predictions",
        "Applicant Risk Scoring",
        "Enter borrower details and receive a probability, score, category, and explanation.",
    )

    applicant = _prediction_form()
    submitted = applicant.pop("_submitted")
    prefer_deployed = applicant.pop("_prefer_deployed")
    active = st.session_state.training_results.get(st.session_state.active_model_name)

    if submitted:
        with st.spinner("Scoring applicant..."):
            prediction = predict_applicant(applicant, active, clean_for_modeling(st.session_state.raw_df), prefer_deployed)
        st.session_state.last_prediction = prediction

    prediction = st.session_state.last_prediction
    if not prediction:
        st.info("Submit the form to generate a prediction.")
        return

    metrics = st.columns(4)
    metrics[0].metric("Default probability", percent(prediction.probability))
    metrics[1].metric("Risk category", prediction.risk_category)
    metrics[2].metric("Credit score", prediction.credit_score)
    metrics[3].metric("Confidence", percent(prediction.confidence))

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.probability_gauge(prediction.probability), use_container_width=True)
    with right:
        st.plotly_chart(charts.contribution_chart(prediction.explanation), use_container_width=True)

    section_title("Prediction Explanation", f"Scored using {prediction.model_name}.")
    if prediction.explanation.empty:
        st.write("A trained session model is required for local feature contribution explanations.")
    else:
        st.dataframe(prediction.explanation, use_container_width=True)
