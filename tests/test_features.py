import numpy as np
import pandas as pd
import pytest

from melbourne_price import config
from melbourne_price.features import MODEL_COLUMNS, engineer_features


def test_matches_processed_dataset():
    """Training data (built in notebook 02) and inference features use the same logic."""
    cleaned = pd.read_csv(config.ROOT / "data/processed/melbourne_housing_cleaned_v1.csv")
    processed = pd.read_csv(config.DATA_PATH)

    engineered = engineer_features(cleaned)

    pd.testing.assert_frame_equal(
        engineered.reset_index(drop=True),
        processed[MODEL_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )


def test_derived_values():
    raw = pd.DataFrame(
        [
            {
                "Suburb": "Richmond",
                "Type": "h",
                "Distance": 2.6,
                "Landsize": 200,
                "BuildingArea": 100,
                "YearBuilt": 1990,
            }
        ]
    )
    out = engineer_features(raw).iloc[0]

    assert out["PropertyAge"] == 27
    assert out["Is_CBD"] == 1
    assert out["Log_Landsize"] == pytest.approx(np.log1p(200))
    assert out["Log_TotalArea"] == pytest.approx(np.log1p(300))


def test_missing_optional_fields_become_nan():
    out = engineer_features(pd.DataFrame([{"Suburb": "Richmond", "Distance": 10}]))
    assert list(out.columns) == MODEL_COLUMNS
    assert out["Is_CBD"].iloc[0] == 0
    assert pd.isna(out["SellerG"].iloc[0])
