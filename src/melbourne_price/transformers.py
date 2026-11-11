"""Custom scikit-learn transformers used in the preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it appears in the training data.

    Categories not seen during ``fit`` are encoded as 0.
    """

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.freq_maps_ = {col: X[col].value_counts().to_dict() for col in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """``SimpleImputer`` wrapper that returns a DataFrame with the original column names."""

    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.imputer_ = SimpleImputer(strategy=self.strategy).fit(X)
        return self

    def transform(self, X):
        X_t = self.imputer_.transform(X)
        return pd.DataFrame(X_t, columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)
