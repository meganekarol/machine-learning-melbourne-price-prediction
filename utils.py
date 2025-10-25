import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
