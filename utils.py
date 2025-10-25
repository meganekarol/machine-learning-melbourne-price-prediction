import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.cols:
            freq_map = X[col].value_counts().to_dict()
            self.freq_maps[col] = freq_map
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Wrapper for SimpleImputer that preserves DataFrame structure"""
    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_t = self.imputer.transform(X)
        return pd.DataFrame(X_t, columns=self.columns_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)
