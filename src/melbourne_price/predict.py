"""Load the trained model and predict prices from raw listing fields."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from melbourne_price import config
from melbourne_price.features import engineer_features


@lru_cache(maxsize=1)
def load_model(path: Path = config.MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m melbourne_price.train` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata(path: Path = config.METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def predict_prices(raw: pd.DataFrame) -> np.ndarray:
    """Return predicted sale prices in AUD for each row of raw listing fields."""
    model = load_model()
    log_pred = model.predict(engineer_features(raw))
    return np.expm1(log_pred)
