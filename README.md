# Deployment Guide (FastAPI)

This guide turns your finished scikit-learn pipeline into a simple, production-friendly HTTP API.
Commands assume you're in the project root (same level as `export_model.py`).

## 0) Prereqs
- Ensure your **custom transformers** (e.g., `FrequencyEncoder`, `DataFrameImputer`) live in a module that's importable.
  Easiest: `utils.py` in the project root; your notebooks/scripts import from it.
- Make sure your **cleaned dataset** exists at `data/melbourne_housing_cleaned_featEngr.csv` (or adjust paths in `export_model.py`).

## 1) Export a fitted pipeline artifact
Loads your tuned pipeline if present, else the base pipeline; fits on train+val (80%), and saves to `artifacts/`.

```bash
python export_model.py
# -> artifacts/model.joblib and artifacts/metadata.json
