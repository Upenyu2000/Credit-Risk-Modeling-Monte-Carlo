# Author: Upenyu Hlangabeza
"""Settings and run instructions page."""

from __future__ import annotations

import streamlit as st

from app.components.layout import page_header, section_title
from app.services.data_service import generate_demo_credit_data, read_uploaded_dataset, summarize_dataset
from app.utils.formatting import compact_number, percent


def render() -> None:
    """Render application settings, data upload, and run guide."""
    page_header(
        "Settings",
        "Workspace Configuration",
        "Load a dataset, reset the demo portfolio, and copy deployment commands.",
    )

    section_title("Dataset Source", "Upload a CSV with a `default` target column, or regenerate the built-in portfolio.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    col1, col2, col3 = st.columns([1, 1, 2])
    if col1.button("Load uploaded CSV", disabled=uploaded is None):
        st.session_state.raw_df = read_uploaded_dataset(uploaded)
        st.session_state.dataset_source = uploaded.name
        st.session_state.training_results = {}
        st.session_state.active_model_name = None
        st.success("Dataset loaded. Retrain models from the Model Training page.")

    rows = col2.number_input("Demo rows", min_value=500, max_value=12000, value=2400, step=100)
    seed = col3.number_input("Demo random seed", min_value=1, max_value=9999, value=42, step=1)
    if st.button("Reset demo dataset"):
        st.session_state.raw_df = generate_demo_credit_data(int(rows), int(seed))
        st.session_state.dataset_source = "Generated demo portfolio"
        st.session_state.training_results = {}
        st.session_state.active_model_name = None
        st.success("Demo dataset regenerated.")

    summary = summarize_dataset(st.session_state.raw_df)
    metrics = st.columns(4)
    metrics[0].metric("Rows", compact_number(summary.rows))
    metrics[1].metric("Columns", compact_number(summary.columns))
    metrics[2].metric("Default rate", percent(summary.default_rate))
    metrics[3].metric("Missing values", compact_number(summary.missing_values))

    section_title("Local Run", "PowerShell commands from the repository root.")
    st.code(
        """python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8501""",
        language="powershell",
    )

    section_title("Web Server Run", "Useful for a remote VM, Codespace, or internal web server.")
    st.code(
        """python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true""",
        language="powershell",
    )

    st.info(
        "For production-style hosting, place the repository on a server, install the requirements in a virtual "
        "environment, expose port 8501 through your firewall or reverse proxy, and run the web-server command."
    )
