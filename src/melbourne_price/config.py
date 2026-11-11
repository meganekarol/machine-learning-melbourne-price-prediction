"""Project paths and constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "melbourne_housing_cleaned_featEngr.csv"
# Overridable so a non-editable install (e.g. Docker) can point at the model files.
MODELS_DIR = Path(os.environ.get("MELBOURNE_MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

TARGET = "Price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
