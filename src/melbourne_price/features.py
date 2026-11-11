"""Feature engineering: turn raw listing fields into the model's input columns.

This mirrors the steps in ``notebooks/02_PreprocessingPipeline.ipynb`` so that
training and inference share exactly one code path.
"""

import numpy as np
import pandas as pd

# Latest construction year in the cleaned dataset (sales span Feb 2016 - Mar 2018).
# PropertyAge is measured relative to this fixed year so that predictions do
# not drift as the calendar moves on.
REFERENCE_YEAR = 2017

# Properties closer than this to the Melbourne CBD are flagged as Is_CBD.
CBD_RADIUS_KM = 5

# Raw fields a caller must (or may) provide.
RAW_COLUMNS = [
    "Suburb",
    "Type",
    "Method",
    "SellerG",
    "Distance",
    "Bedroom",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "CouncilArea",
    "Latitude",
    "Longtitude",  # spelling as in the source dataset
    "Regionname",
    "Propertycount",
]

FREQ_COLS = ["Suburb", "SellerG", "CouncilArea"]
ONEHOT_COLS = ["Type", "Method", "Regionname"]
NUMERIC_COLS = [
    "Distance",
    "Bedroom",
    "Bathroom",
    "Car",
    "Propertycount",
    "Latitude",
    "Longtitude",
    "PropertyAge",
    "Log_Landsize",
    "Log_BuildingArea",
    "Log_TotalArea",
    "Is_CBD",
]

# Column order of data/processed/melbourne_housing_cleaned_featEngr.csv (minus Price).
MODEL_COLUMNS = [
    "Suburb",
    "Type",
    "Method",
    "SellerG",
    "Distance",
    "Bedroom",
    "Bathroom",
    "Car",
    "CouncilArea",
    "Latitude",
    "Longtitude",
    "Regionname",
    "Propertycount",
    "Log_Landsize",
    "Log_BuildingArea",
    "PropertyAge",
    "Log_TotalArea",
    "Is_CBD",
]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the engineered features from raw listing fields.

    Missing optional fields are added as NaN; the pipeline's imputers fill them.
    """
    df = raw.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric = [
        "Distance",
        "Bedroom",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Latitude",
        "Longtitude",
        "Propertycount",
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # None -> NaN so the pipeline's imputers recognise missing categorical values
    categorical = FREQ_COLS + ONEHOT_COLS
    df[categorical] = df[categorical].astype(object).where(df[categorical].notna(), np.nan)

    df["Log_Landsize"] = np.log1p(df["Landsize"])
    df["Log_BuildingArea"] = np.log1p(df["BuildingArea"])
    df["PropertyAge"] = REFERENCE_YEAR - df["YearBuilt"]
    df["Log_TotalArea"] = np.log1p(df["Landsize"] + df["BuildingArea"])
    df["Is_CBD"] = (df["Distance"] < CBD_RADIUS_KM).astype(int)

    return df[MODEL_COLUMNS]
