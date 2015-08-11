import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class averager(BaseEstimator, TransformerMixin):
    """Average data set into n samples"""
    def __init__(self, n, mean=None):
        if mean is None:
            mean = np.mean
        self.n = int(n)
        self.mean = mean

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        if len(X) <= self.n:
            return X
        Xt = np.zeros(self.n, X.shape[1:])
        y = np.round(np.arange(len(X)) / float(self.n))
        for ii in range(n):
            sel = np.where(y == ii)[0]
            Xt[ii, ...] = self.mean(X[sel, ...], axis=0)
        return Xt
