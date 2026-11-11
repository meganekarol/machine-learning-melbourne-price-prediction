"""Train the final model and write it to ``models/``.

Usage:
    python -m melbourne_price.train

Reproduces notebooks/04_EvaluationAnalysis.ipynb: 80/20 train/test split,
5-fold CV on the training pool, fit on the training pool, evaluate on test.
"""

import json
from datetime import UTC, datetime
from math import sqrt

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from melbourne_price import config
from melbourne_price.features import MODEL_COLUMNS, RAW_COLUMNS, REFERENCE_YEAR
from melbourne_price.pipeline import GB_BEST_PARAMS, build_final_pipeline


def evaluate(y_true_log, y_pred_log) -> dict:
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "r2_log": r2_score(y_true_log, y_pred_log),
        "mae_log": mean_absolute_error(y_true_log, y_pred_log),
        "rmse_log": sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "r2_aud": r2_score(y_true, y_pred),
        "mae_aud": mean_absolute_error(y_true, y_pred),
        "rmse_aud": sqrt(mean_squared_error(y_true, y_pred)),
    }


def main() -> dict:
    df = pd.read_csv(config.DATA_PATH)
    X = df[MODEL_COLUMNS]
    y = np.log1p(df[config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    pipe = build_final_pipeline()

    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    cv_r2 = cv_scores["test_score"]
    print(f"CV R2 (log): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    pipe.fit(X_train, y_train)
    test_metrics = evaluate(y_test, pipe.predict(X_test))
    for k, v in test_metrics.items():
        print(f"test {k:>9}: {v:,.4f}")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_type": "GradientBoostingRegressor",
        "hyperparameters": GB_BEST_PARAMS,
        "target": "log1p(Price)",
        "currency": "AUD",
        "raw_columns": RAW_COLUMNS,
        "model_columns": MODEL_COLUMNS,
        "reference_year": REFERENCE_YEAR,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_r2_log_mean": float(cv_scores["test_score"].mean()),
        "cv_r2_log_std": float(cv_scores["test_score"].std()),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "sklearn_version": sklearn.__version__,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model and metadata to {config.MODELS_DIR}")
    return metadata


if __name__ == "__main__":
    main()
