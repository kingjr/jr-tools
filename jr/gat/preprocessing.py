import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Averager(BaseEstimator, TransformerMixin):
    """Average data set into n samples"""
    def __init__(self, n, mean=None):
        if mean is None:
            mean = np.mean
        self.n = int(n)
        self.mean = mean

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X) <= self.n:
            return X
        Xt = np.zeros((self.n, X.shape[1]))
        y = np.round(np.arange(len(X)) / float(len(X)) * self.n)
        self.y_ = y
        for ii in range(self.n):
            sel = np.where(y == ii)[0]
            Xt[ii, ...] = self.mean(X[sel, ...], axis=0)
        return Xt


class MeanFeatures(BaseEstimator, TransformerMixin):
    """Reshape and mean data along given axis."""
    def __init__(self, shape, axis):
        self.axis = axis
        self.shape = shape

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        X = np.reshape(X, np.hstack((X.shape[0], self.shape)))
        return np.mean(X, axis=self.axis)


class DigitizedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, estimator):
        self.bins = bins
        self.estimator = estimator

    def fit(self, X, y):
        y_bin = np.digitize(y, self.bins)
        self.estimator.fit(X, y=y_bin)
        return self

    def transform(self, X):
        return self.estimator.transform(X)


class GenericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, function, **fit_params):
        self.function = function
        self.fit_params = fit_params

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return self.function(X, **self.fit_params)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
