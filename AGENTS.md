# AGENTS.md

Author: Upenyu Hlangabeza

Guidance for coding agents working in this repository.

## Project Overview

This repository contains an end-to-end credit risk modeling project. The main interactive application is the root Streamlit app in `app.py`, with modular UI, service, model, and utility code under `app/`. The original simple Streamlit predictor remains in `project-root/` and loads a serialized XGBoost model plus preprocessing objects from `project-root/model/model_data.pkl`.

Top-level areas:

- `project-root/`: Streamlit application, prediction utilities, requirements, and model artifacts.
- `app.py`: root entry point for the interactive Credit Risk Analysis Studio.
- `app/`: modular GUI package with pages, components, services, schemas, utilities, and assets.
- `notebooks/`: exploratory analysis, data cleaning, feature engineering, model building, and evaluation notebooks.
- `finding-documents/`: presentation and model evaluation report artifacts.
- `images/`: screenshots and model interpretation images used by the README.

## Setup and Run

Use Python 3.8 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app normally runs at `http://localhost:8501/`. See `RUN_APP.md` for local and web-server deployment commands.

## Important Path Notes

The current code uses paths such as `project-root/model/model_data.pkl` and `project-root/Lauki Finance.JPG`. These paths assume commands are run from the repository root. If you run commands from inside `project-root/`, update the path handling or set the working directory appropriately.

When changing path logic, prefer robust paths relative to the Python file location, for example `Path(__file__).resolve().parent`, instead of relying on the process working directory.

## Code Guidelines

- Keep the new GUI entry point in `app.py`.
- Keep page-specific UI in `app/pages/`.
- Keep reusable UI components and chart factories in `app/components/`.
- Keep data, preprocessing, training, prediction, and explainability logic in `app/services/`.
- Keep shared dataclasses in `app/models/`.
- Keep utility helpers in `app/utils/`.
- Keep legacy predictor changes in `project-root/main.py` and `project-root/utils.py`.
- Avoid changing serialized model files in `project-root/model/` unless the task explicitly involves retraining or replacing the model.
- Preserve the feature order expected by `model_data.pkl`; prediction inputs must match `model_data["features"]`.
- Keep notebook changes focused. Do not rewrite generated notebook metadata unless necessary.
- Use ASCII in new code unless editing text that already requires non-ASCII.

## Verification

There is no dedicated automated test suite in the repository. For code changes, use the narrowest practical checks:

```powershell
python -m compileall app.py app project-root
```

For app behavior changes, run:

```powershell
streamlit run app.py
```

Then verify that the app loads, accepts borrower inputs, and returns default probability, credit score, and rating after clicking `Calculate Risk`.

## Dependency Notes

Dependencies are pinned in `project-root/requirements.txt`. Do not add dependencies casually. If a new package is required, update the requirements file and document why it is needed.

## Git Hygiene

- Do not revert unrelated user changes.
- Keep edits scoped to the requested task.
- Treat model binaries, PDFs, PPTX files, and images as project artifacts; avoid regenerating or replacing them unless requested.
