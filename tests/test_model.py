import pandas as pd

from melbourne_price import config
from melbourne_price.predict import load_metadata, predict_prices


def test_metadata_metrics_meet_baseline():
    """Guard against regressions vs. the notebook 04 results (test R2 on log scale ~0.78)."""
    metrics = load_metadata()["test_metrics"]
    assert metrics["r2_log"] > 0.75
    assert metrics["mae_aud"] < 200_000


def test_predictions_are_positive_and_plausible():
    raw = pd.read_csv(config.ROOT / "data/processed/melbourne_housing_cleaned_v1.csv").head(50)
    prices = predict_prices(raw)
    assert (prices > 0).all()
    # Median prediction should be in the same ballpark as the median actual price
    ratio = pd.Series(prices).median() / raw["Price"].median()
    assert 0.7 < ratio < 1.3
