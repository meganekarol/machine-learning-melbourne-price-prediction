"""
Export the final fitted pipeline for deployment.
This script loads your final GBR pipeline and saves it in a standardized location:
    artifacts/model.joblib  and  artifacts/metadata.json
"""
import os, json, joblib, pandas as pd, numpy as np
from datetime import datetime

FINAL_PIPELINE_PATH = "artifacts/melbourne_pipeline_final.pkl"
DATA_PATH = "data/melbourne_housing_cleaned_featEngr.csv"
OUTPUT_MODEL = "artifacts/model.joblib"
OUTPUT_META = "artifacts/metadata.json"

def main():
    if not os.path.exists(FINAL_PIPELINE_PATH):
        raise FileNotFoundError(f"Could not find final pipeline at {FINAL_PIPELINE_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    print("Loading final pipeline...")
    pipeline = joblib.load(FINAL_PIPELINE_PATH)
    print("Loaded pipeline from:", FINAL_PIPELINE_PATH)

    # Load data only to extract column names
    df = pd.read_csv(DATA_PATH)
    if "Price" not in df.columns:
        raise ValueError("Expected a 'Price' column in dataset.")
    X = df.drop(columns=["Price"])

    # Save standardized model artifact
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, OUTPUT_MODEL)

    # Metadata for deployment
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_pipeline": FINAL_PIPELINE_PATH,
        "n_features": len(X.columns),
        "columns": list(X.columns),
        "target": "log1p(Price)",
        "framework": "scikit-learn",
        "model_type": "GradientBoostingRegressor",
        "random_state": 42
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Export complete:\n  - {OUTPUT_MODEL}\n  - {OUTPUT_META}")

if __name__ == "__main__":
    main()
