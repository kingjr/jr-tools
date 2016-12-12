import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from mne.filter import low_pass_filter, high_pass_filter, band_pass_filter
from mne.parallel import parallel_func

from nose.tools import assert_true


class _BaseEstimator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def baseline(X, mode, tslice):
        if X.shape[-1] > 0:
            mean = np.mean(X[..., tslice], axis=-1)[..., None]
        else:
            mean = 0  # otherwise we get an ugly nan
        if mode == 'mean':
            X -= mean
        if mode == 'logratio':
            X /= mean
            X = np.log10(X)  # a value of 1 means 10 times bigger
        if mode == 'ratio':
            X /= mean
        elif mode == 'zscore':
            std = np.std(X[..., tslice], axis=-1)[..., None]
            X -= mean
            X /= std
        elif mode == 'percent':
            X -= mean
            X /= mean
        elif mode == 'zlogratio':
            X /= mean
            X = np.log10(X)
            std = np.std(X[..., tslice], axis=-1)[..., None]
            X /= std
        return X


class EpochsBaseliner(_BaseEstimator):
    def __init__(self, tslice=None, mode='mean'):
        self.mode = mode
        self.tslice = slice(None) if tslice is None else tslice
        assert_true(self.mode in ['mean', 'logratio', 'ratio', 'zscore',
                                  'percent', 'zlogratio'])
        assert_true(isinstance(self.tslice, (slice, int)))

    def transform(self, X):
        return baseline(X, self.mode, self.tslice)


class TimeFreqBaseliner(_BaseEstimator):
    def __init__(self, tslice=None, mode='mean'):
        self.mode = mode
        self.tslice = slice(None) if tslice is None else tslice
        assert_true(self.mode in ['mean', 'logratio', 'ratio', 'zscore',
                                  'percent', 'zlogratio'])

    def transform(self, X):
        return baseline(X, self.mode, self.tslice)


class TimePadder(_BaseEstimator):
    """Padd time before and after epochs"""
    def __init__(self, n_sample, value=0.):
        self.n_sample = n_sample
        assert_true(isinstance(self.n_sample, int))
        self.value = value
        assert_true(isinstance(value, (int, float)) or (value == 'median'))

    def transform(self, X):
        if self.value == 'median':
            coefs = np.median(X, axis=2)
        else:
            coefs = self.value * np.ones(X.shape[:2])
        coefs = np.tile(coefs, [self.n_sample, 1, 1]).transpose([1, 2, 0])
        X = np.concatenate((coefs, X, coefs), axis=2)
        return X

    def inverse_transform(self, X):
        X = X[:, :, self.n_sample:-self.n_sample]
        return X


class TimeSelector(_BaseEstimator):
    """Padd time before and after epochs"""
    def __init__(self, tslice):
        self.tslice = tslice
        assert_true(isinstance(self.tslice, (slice, int)))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = X[:, :, self.tslice]
        return X


class TimeFreqSelector(_BaseEstimator):
    """Padd time before and after epochs"""
    def __init__(self, tslice=None, fslice=None):
        self.tslice = slice(None) if tslice is None else tslice
        self.fslice = slice(None) if fslice is None else fslice
        assert_true(isinstance(self.tslice, (slice, int)))
        assert_true(isinstance(self.fslice, (slice, int)))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = X[:, :, :, self.tslice]
        X = X[:, :, self.fslice, :]
        return X


class MyXDawn(_BaseEstimator):
    """Wrapper for pyriemann Xdawn + robust.
    Will eventually need to clean both MNE and pyriemann with refactorings"""

    def __init__(self, n_filter=4, estimator='scm'):
        from pyriemann.estimation import Xdawn
        self.n_filter = n_filter
        assert_true(isinstance(self.n_filter, int))
        self.estimator = estimator
        assert_true(isinstance(estimator, str))
        self._xdawn = Xdawn(nfilter=n_filter, estimator=estimator)

    def fit(self, X, y):
        # only apply on channels who std > 0 across time on at least one trial
        self.picks_ = np.where(np.mean(np.std(X, axis=2) ** 2, axis=0))[0]
        self._xdawn.fit(X[:, self.picks_, :], y)
        return self

    def transform(self, X):
        return self._xdawn.transform(X[:, self.picks_, :])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class SpatialFilter(_BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        assert_true(isinstance(estimator, TransformerMixin))

    def fit(self, X, y=None):
        n_epoch, n_chan, n_time = X.shape
        # trial as time
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        self.estimator.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        n_epoch, n_chan, n_time = X.shape
        # trial as time
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        X = self.estimator.transform(X)
        X = np.reshape(X.T, [-1, n_epoch, n_time]).transpose([1, 0, 2])
        return X


class Reshaper(_BaseEstimator):
    """Transpose, concatenate and/or reshape data.

    Parameters
    ----------
    concatenate : int | None
        Reshaping feature dimension e.g. np.concatenate(X, axis=concatenate).
        Defaults to None.
    transpose : array of int, shape(1 + n_dims) | None
        Reshaping feature dimension e.g. X.transpose(transpose).
        Defaults to None.
    reshape : array, shape(n_dims) | None
        Reshaping feature dimension e.g. X.reshape(np.r_[len(X), shape]).
        Defaults to -1 if concatenate or transpose is None, else defaults
        to None.

    """

    def __init__(self, reshape=None, transpose=None, concatenate=None,
                 verbose=False):
        if (reshape is None) and (transpose is None) and (concatenate is None):
            reshape = [-1]
        self.reshape = reshape
        self.transpose = transpose
        self.concatenate = concatenate
        self.verbose = verbose

    def fit(self, X, y=None):
        self.shape_ = X.shape[1:]
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X, y=None):
        if self.transpose is not None:
            X = X.transpose(self.transpose)
        if self.concatenate:
            X = np.concatenate(X, self.concatenate)
        if self.reshape is not None:
            X = np.reshape(X, np.hstack((X.shape[0], self.reshape)))
        if self.verbose:
            print(self.shape_, '->', (X.shape[1:]))
        return X


class LightTimeDecoding(_BaseEstimator):
    def __init__(self, estimator=None, method='predict', n_jobs=1):
        self.estimator = (LogisticRegression() if estimator is None
                          else estimator)
        self.method = method
        assert_true(self.method in ['predict', 'predict_proba'])
        assert_true(hasattr(self.estimator, method))
        self.n_jobs = n_jobs
        assert_true(isinstance(self.n_jobs, int))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def fit(self, X, y):
        self.estimators_ = list()
        parallel, p_func, n_jobs = parallel_func(_fit, self.n_jobs)
        estimators = parallel(
            p_func(self.estimator, split, y)
            for split in np.array_split(X, n_jobs, axis=2))
        self.estimators_ = np.concatenate(estimators, 0)
        return self

    def transform(self, X):
        parallel, p_func, n_jobs = parallel_func(_predict_decod, self.n_jobs)
        X_splits = np.array_split(X, n_jobs, axis=2)
        est_splits = np.array_split(self.estimators_, n_jobs)
        y_pred = parallel(
            p_func(est_split, x_split, self.method)
            for (est_split, x_split) in zip(est_splits, X_splits))

        if n_jobs > 1:
            y_pred = np.concatenate(y_pred, axis=1)
        else:
            y_pred = y_pred[0]
        return y_pred

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)


def _fit(estimator, X, y):
    estimators_ = list()
    for ii in range(X.shape[2]):
        est = clone(estimator)
        est.fit(X[:, :, ii], y)
        estimators_.append(est)
    return estimators_


def _predict_decod(estimators, X, method):
    n_sample, n_chan, n_time = X.shape
    y_pred = np.array((n_sample, n_time))
    for ii, est in enumerate(estimators):
        if method == 'predict':
            _y_pred = est.predict(X[:, :, ii])
        elif method == 'predict_proba':
            _y_pred = est.predict_proba(X[:, :, ii])
        # init
        if ii == 0:
            y_pred = _init_pred(_y_pred, X)
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _init_pred(y_pred, X):
    n_sample, n_chan, n_time = X.shape
    if y_pred.ndim == 2:
        y_pred = np.zeros((n_sample, n_time, y_pred.shape[-1]))
    else:
        y_pred = np.zeros((n_sample, n_time))
    return y_pred


class LightGAT(LightTimeDecoding):
    def transform(self, X):
        parallel, p_func, n_jobs = parallel_func(_predict_gat, self.n_jobs)
        y_pred = parallel(
            p_func(self.estimators_, x_split, self.method)
            for x_split in np.array_split(X, n_jobs, axis=2))

        y_pred = np.concatenate(y_pred, axis=2)
        return y_pred


def _predict_gat(estimators, X, method):
    n_sample, n_chan, n_time = X.shape
    for ii, est in enumerate(estimators):
        X_stack = np.transpose(X, [1, 0, 2])
        X_stack = np.reshape(X_stack, [n_chan, n_sample * n_time]).T
        if method == 'predict':
            _y_pred = est.predict(X_stack)
            _y_pred = np.reshape(_y_pred, [n_sample, n_time])
        elif method == 'predict_proba':
            _y_pred = est.predict_proba(X_stack)
            n_dim = _y_pred.shape[-1]
            _y_pred = np.reshape(_y_pred, [n_sample, n_time, n_dim])
        # init
        if ii == 0:
            y_pred = _init_pred_gat(_y_pred, X, len(estimators))
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _init_pred_gat(y_pred, X, n_train):
    n_sample, n_chan, n_time = X.shape
    if y_pred.ndim == 3:
        y_pred = np.zeros((n_sample, n_train, n_time, y_pred.shape[-1]))
    else:
        y_pred = np.zeros((n_sample, n_train, n_time))
    return y_pred


class CustomEnsemble(TransformerMixin):
    def __init__(self, estimators, method='predict'):
        self.estimators = estimators
        self.method = method
        assert_true(method in ['predict', 'predict_proba'])

    def fit(self, X, y=None):
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        all_Xt = list()
        for estimator in self.estimators:
            if self.method == 'predict':
                Xt = estimator.predict(X)
            elif self.method == 'predict_proba':
                Xt = estimator.predict_proba(X)
            all_Xt.append(Xt)
        all_Xt = np.c_[all_Xt].T
        return all_Xt

    def get_params(self, deep=True):
        return dict(estimators=self.estimators, method=self.method)


class GenericTransformer(_BaseEstimator):
    def __init__(self, function, **fit_params):
        self.function = function
        self.fit_params = fit_params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.function(X, **self.fit_params)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)


class Filterer(_BaseEstimator):
    def __init__(self, sfreq, l_freq=None, h_freq=None, filter_length='10s',
                 l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
                 method='fft', iir_params=None):
        self.sfreq = sfreq
        self.l_freq = None if l_freq == 0 else l_freq
        self.h_freq = None if h_freq > (sfreq / 2.) else h_freq
        if (self.l_freq is not None) and (self.h_freq is not None):
            assert_true(self.l_freq < self.h_freq)
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.n_jobs = n_jobs
        self.method = method
        self.iir_params = iir_params
        assert_true((l_freq is None) or isinstance(l_freq, (int, float)))
        assert_true((h_freq is None) or isinstance(h_freq, (int, float)))

    def transform(self, X, y=None):

        kwargs = dict(Fs=self.sfreq, filter_length=self.filter_length,
                      method=self.method, iir_params=self.iir_params,
                      copy=False, verbose=False, n_jobs=self.n_jobs)
        if self.l_freq is None and self.h_freq is not None:
            filter_func = low_pass_filter
            kwargs['Fp'] = self.h_freq
            kwargs['trans_bandwidth'] = self.h_trans_bandwidth

        if self.l_freq is not None and self.h_freq is None:
            filter_func = high_pass_filter
            kwargs['Fp'] = self.l_freq
            kwargs['trans_bandwidth'] = self.l_trans_bandwidth

        if self.l_freq is not None and self.h_freq is not None:
            filter_func = band_pass_filter
            kwargs['Fp1'] = self.l_freq
            kwargs['Fp2'] = self.h_freq
            kwargs['l_trans_bandwidth'] = self.l_trans_bandwidth
            kwargs['h_trans_bandwidth'] = self.h_trans_bandwidth

        return filter_func(X, **kwargs)


class TimeEmbedder(_BaseEstimator):
    def __init__(self, delays=2):
        self.delays = delays

    def transform(self, X, y=None):
        if not isinstance(X, np.ndarray):
            epochs = X
            X = epochs._data

        if isinstance(self.delays, int):
            delays = range(1, self.delays)
        else:
            delays = self.delays

        X2 = []
        for x in X:
            tmp = x
            for d in delays:
                tmp = np.r_[tmp, np.roll(x, d, axis=-1)]
            X2.append(tmp)
        X2 = np.array(X2)
        return X2

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X, y)


class Windower(TransformerMixin, BaseEstimator):
    """To make sliding windows

    Parameters
    ----------
    size : int
        The window size.
    step : int
        The window step.
    vectorize : bool
        Returns arrays or vector.
    """
    def __init__(self, size=1, step=1, vectorize=False):
        self.size = size
        self.step = step
        self.vectorize = vectorize

    def fit(self, X, y=None):
        """Does nothing, for sklearn compatibility purposes

        Parameters
        ----------
        X : ndarray, shape(n_epochs, n_times, n_features)
            The target data.
        y : None | array, shape(n_epochs,)

        Returns
        -------
        self : self
        """
        if X.ndim != 3:
            raise ValueError('expects 3D array')
        return self

    def transform(self, X, y=None):
        """Generate windows from X.

        Parameters
        ----------
        X : ndarray, shape(n_epochs, n_times, n_features)
            The target data.
        y : None | array, shape(n_epochs,)

        Returns
        -------
        Xt : ndarray, shape(n_epochs, n_features, n_window_times, n_windows)
            The transformed data. If vectorize is True, then shape is
            (n_epochs, -1).
        """
        Xt = list()
        for time in range(0, X.shape[2] - self.size, self.step):
            Xt.append(X[:, :, time:(time + self.size)])
        Xt = np.transpose(Xt, [1, 2, 3, 0])  # trial chan window time
        if self.vectorize:
            Xt = Xt.reshape([len(Xt), -1, Xt.shape[-1]])
        return Xt

    def fit_transform(self, X, y=None):
        """Generate windows from X.

        Parameters
        ----------
        X : ndarray, shape(n_epochs, n_times, n_features)
            The target data.
        y : None | array, shape(n_epochs,)

        Returns
        -------
        Xt : ndarray, shape(n_epochs, n_features, n_window_times, n_windows)
            The transformed data. If vectorize is True, then shape is
            (n_epochs, -1).
        """
        return self.fit(X).transform(X)


def test_windower():
    Windower(3, 2, False).transform(np.zeros((2, 30, 100))).shape
