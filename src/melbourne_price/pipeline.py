"""Construction of the scikit-learn preprocessing + model pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from melbourne_price.features import FREQ_COLS, NUMERIC_COLS, ONEHOT_COLS
from melbourne_price.transformers import DataFrameImputer, FrequencyEncoder

# Best hyperparameters found by GridSearchCV in notebooks/03_ModelSelection.ipynb
GB_BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_samples_split": 5,
    "n_estimators": 150,
    "random_state": 0,
}


def build_preprocessor() -> ColumnTransformer:
    freq_pipe = Pipeline(
        [
            ("imputer", DataFrameImputer(strategy="most_frequent")),
            ("freq_enc", FrequencyEncoder(cols=FREQ_COLS)),
        ]
    )
    onehot_pipe = Pipeline(
        [
            ("imputer", DataFrameImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        [
            ("imputer", DataFrameImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("freq", freq_pipe, FREQ_COLS),
            ("onehot", onehot_pipe, ONEHOT_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )


def build_base_pipeline() -> Pipeline:
    """Base pipeline used for model comparison (notebooks 02 and 03).

    Step names ``variance``, ``rfecv`` and ``model`` can be swapped with ``set_params``.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("rfecv", "passthrough"),
            ("model", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)),
        ]
    )


def build_final_pipeline() -> Pipeline:
    """Final tuned Gradient Boosting pipeline (notebook 04 / production)."""
    pipe = build_base_pipeline()
    pipe.set_params(
        variance="passthrough",
        model=GradientBoostingRegressor(**GB_BEST_PARAMS),
    )
    return pipe
