from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import joblib, json, os
import numpy as np

APP_TITLE = "Melbourne Housing Price Predictor"
APP_VERSION = "1.1.0"

ART_MODEL = os.path.join(os.path.dirname(__file__), "artifacts", "model.joblib")
ART_META  = os.path.join(os.path.dirname(__file__), "artifacts", "metadata.json")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Load model + metadata once, on startup
if not os.path.exists(ART_MODEL):
    raise RuntimeError("Model artifact not found at " + ART_MODEL + ". Run export_model.py first.")
model = joblib.load(ART_MODEL)

if os.path.exists(ART_META):
    with open(ART_META, "r") as f:
        META = json.load(f)
else:
    META = {"columns": None, "target": None}

EXPECTED_COLUMNS: Optional[List[str]] = META.get("columns")

# --- auto-detect target transform ------------------------------------------------
target_desc = (META.get("target") or "").lower()
# "log1p(Price)" or similar → treat as log1p
USE_LOG1P = "log1p" in target_desc

class Item(BaseModel):
    features: Dict[str, Any]

class Batch(BaseModel):
    items: List[Item]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "model_info": META}

def _df_from_payload(items: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([it["features"] for it in items])
    if EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
        # Extra columns OK: they’ll be ignored by the ColumnTransformer if unused
        df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=None)
    return df

def _predict_core(df: pd.DataFrame, return_scale: Literal["eur","log","both"]) -> Dict[str, Any]:
    preds = model.predict(df)

    if USE_LOG1P:
        # model outputs log1p(price) → back-transform
        preds_eur = np.expm1(preds)
        if return_scale == "eur":
            return {"predictions_eur": preds_eur.tolist(), "target_transform": "log1p→eur"}
        elif return_scale == "log":
            return {"predictions_log": preds.tolist(), "target_transform": "log1p"}
        else:  # both
            return {
                "predictions_log": preds.tolist(),
                "predictions_eur": preds_eur.tolist(),
                "target_transform": "log1p→eur"
            }
    else:
        # model already predicts in euros
        if return_scale == "log":
            raise HTTPException(status_code=400, detail="Model does not use a log target; 'log' not available.")
        # eur or both → only eur is meaningful
        return {"predictions_eur": preds.tolist(), "target_transform": "none"}

@app.post("/predict")
def predict(
    batch: Batch,
    return_scale: Literal["eur","log","both"] = Query("eur", description="Return predictions in 'eur', 'log', or 'both'")
):
    try:
        df = _df_from_payload([it.dict() for it in batch.items])
        return _predict_core(df, return_scale)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_one")
def predict_one(
    item: Item,
    return_scale: Literal["eur","log","both"] = Query("eur", description="Return predictions in 'eur', 'log', or 'both'")
):
    return predict(Batch(items=[item]), return_scale=return_scale)
