# Credit Risk Analysis Studio

Author: Upenyu Hlangabeza

This repository now includes a modular Streamlit GUI for exploring the full credit-risk machine learning workflow: dataset browsing, cleaning, EDA, feature engineering, model training, evaluation, prediction, explainability, and settings.

## Local Run

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## Web Server Run

Use this on a VM, remote workstation, container, or internal server:

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Expose port `8501` through your firewall or reverse proxy. For HTTPS, put Streamlit behind Nginx, Caddy, Traefik, or your platform's load balancer.

## Streamlit Community Cloud

Use:

- Main file path: `app.py`
- Python dependencies: `requirements.txt`
- Python version: 3.10 or newer

## Notes

- If no CSV is uploaded, the app generates a realistic demo loan portfolio that follows the feature contract described by the notebooks and model artifact.
- The prediction page can use the original serialized model artifact when its dependencies are installed.
- The explainability page uses model-agnostic perturbation analysis rather than adding a heavy SHAP dependency.
