# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sklearn.preprocessing import StandardScaler


def _stand_mad(a, median):
    """ Fast sandard MAD
    Parameters
    ----------
    a : np.array, shape(n_samples, n_dims)
    median : np.array, shape(n_dims)

    Returns
    -------
    mad : np.array, shape(n_dims)
    Adapted from based on statsmodel.robust.scale.stand_mad"""
    from scipy.stats import norm
    from statsmodels.tools import tools
    c = norm.ppf(3/4.)
    a = np.asarray(a)
    d = tools.unsqueeze(median, 0, a.shape)
    return np.median(np.fabs(a - d)/c, axis=0)


class StandardScaler32(StandardScaler):
    """Identical to StandardScaler but allows 'float_precision' parameter
    in init' to minimize memory storage. Precision is changed after fitting.
    """

    def fit(self, X, y=None):
        super(StandardScaler32, self).fit(X, y=None)
        self.mean_ = np.float32(self.mean_)
        self.std_ = np.float32(self.std_)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class MedianScaler(StandardScaler):
    """Similar to sklearn.preprocessing.StandardScaler based on medians"""

    def fit(self, X, y=None):
        self.mean_ = np.median(X, axis=0)
        self.std_ = _stand_mad(X, self.mean_)
        self.std_[self.std_ == 0.0] = 1.0
        return self


class MedianClassScaler(StandardScaler):
    """Similar to sklearn.preprocessing.StandardScaler based on medians
    computed on each class separately"""
    def fit(self, X, y):
        for i, yclass in enumerate(np.unique(y)):
            # find corresponding samples
            sel = np.where(y == yclass)[0]
            # compute and store median
            mean = np.median(X[sel, :], axis=0)
            # compute and store mad
            std = _stand_mad(X[sel, :], mean)
            std[std == 0.0] = 1.0
            if i == 0:
                means = mean
                stds = std
            else:
                means = np.vstack((means, mean))
                stds = np.vstack((stds, std))
        # mean params across classes
        self.mean_ = np.mean(means, axis=0)
        self.std_ = np.mean(stds, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self


class StandardClassScaler(StandardScaler):
    """Similar to sklearn.preprocessing.StandardScaler computed on each class
    separately"""
    def fit(self, X, y):
        import numpy as np
        for i, yclass in enumerate(np.unique(y)):
            sel = np.where(y == yclass)[0]
            scaler_ = StandardScaler()
            scaler_.fit(X[sel, :])
            mean = scaler_.mean_
            std = scaler_.std_
            if i == 0:
                means = mean
                stds = std
            else:
                means = np.vstack((means, mean))
                stds = np.vstack((stds, std))
        # mean params across classes
        self.mean_ = np.mean(means, axis=0)
        self.std_ = np.mean(stds, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self
