import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.time_frequency import single_trial_power
from mne.parallel import parallel_func
from pyriemann.estimation import Xdawn


class EpochsTransformerMixin(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        # implement fit to allow debugging
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _reshape(self, X):
        # Recontruct epochs
        n_epoch = len(X)
        n_chan = len(self.info['chs'])
        n_time = np.prod(X.shape[1:]) // n_chan
        X = np.reshape(X, [n_epoch, n_chan, n_time])
        return X


class TimeFreqTransformerMixin(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        # implement fit to allow debugging
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def _reshape(self, X):
        # Recontruct time freq epochs
        n_epoch = len(X)
        n_chan = len(self.info['chs'])
        n_freq = len(self.frequencies)
        n_time = np.prod(X.shape[1:]) // (n_chan * n_freq)
        X = np.reshape(X, [n_epoch, n_chan, n_freq, n_time])
        return X


class Baseliner(EpochsTransformerMixin):
    def __init__(self, info, scaler=None, tslice=None):
        self.info = info
        self.scaler = (StandardScaler() if scaler is None else scaler)
        self.tslice = slice(None) if tslice is None else tslice

    def transform(self, X):
        # reshape epochs
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape

        def time_transpose(X):
            # For each sample in each channel: fit across time points
            return X.reshape([-1, X.shape[2]]).T

        # Fit on selected region (e.g. baseline)
        self.scaler.fit(time_transpose(X[:, :, self.tslice]))

        # Apply across all chan & samples
        X = self.scaler.transform(time_transpose(X))

        # epoch transpose
        X = X.T.reshape([n_epoch, n_chan, n_time])
        return X


class TimeFreqDecomposer(EpochsTransformerMixin):
    def __init__(self,  info, frequencies, use_fft=True, n_cycles=7,
                 baseline=None, baseline_mode='ratio', times=None,
                 decim=1, n_jobs=1, zero_mean=False, verbose=None):
        self.info = info
        self.frequencies = frequencies
        self.use_fft = use_fft
        self.n_cycles = n_cycles
        self.baseline = baseline
        self.baseline_mode = baseline_mode
        self.times = times
        self.decim = decim
        self.n_jobs = n_jobs
        self.zero_mean = zero_mean
        self.verbose = verbose

    def transform(self, X):
        sfreq = self.info['sfreq']

        # Recontruct epochs
        X = self._reshape(X)

        # Time Frequency decomposition
        tfr = single_trial_power(
            X, sfreq=sfreq, frequencies=self.frequencies, use_fft=self.use_fft,
            n_cycles=self.n_cycles, baseline=self.baseline,
            baseline_mode=self.baseline_mode, times=self.times,
            decim=self.decim, n_jobs=self.n_jobs, zero_mean=self.zero_mean,
            verbose=self.verbose)

        return tfr


class TimePadder(EpochsTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, info, n_sample, value=0):
        self.n_sample = n_sample
        self.info = info
        self.value = value

    def transform(self, X):
        X = self._reshape(X)
        if self.value == 'median':
            coefs = np.median(X, axis=2)
        else:
            coefs = np.zeros(X.shape[:2])
        coefs = np.tile(coefs, [self.n_sample, 1, 1]).transpose([1, 2, 0])
        X = np.concatenate((coefs, X, coefs), axis=2)
        return X

    def inverse_transform(self, X):
        X = self._reshape(X)
        X = X[:, :, self.n_sample:-self.n_sample]
        return X.reshape([len(X), -1])


class TimeCropper(EpochsTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, info, tslice=None):
        self.tslice = slice(None) if tslice is None else tslice
        self.info = info

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        X = X[:, :, self.tslice]
        return X.reshape([len(X), -1])


class TimeFreqCropper(TimeFreqTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, info, frequencies, tslice=None, fslice=None):
        self.tslice = slice(None) if tslice is None else tslice
        self.fslice = slice(None) if fslice is None else fslice
        self.info = info
        self.frequencies = frequencies

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        X = X[:, :, :, self.tslice]
        X = X[:, :, self.fslice, :]
        return X.reshape([len(X), -1])


class MyXDawn(EpochsTransformerMixin):
    """Wrapper for pyriemann Xdawn + robust.
    Will eventually need to clean both MNE and pyriemann with refactorings"""

    def __init__(self, info, n_filter=4, estimator='scm'):
        self.info = info
        self.n_filter = n_filter
        self.estimator = estimator
        self._xdawn = Xdawn(nfilter=n_filter, estimator=estimator)

    def fit(self, X, y):
        X = self._reshape(X)
        # only apply on channels who std > 0 across time on at least one trial
        self.picks_ = np.where(np.mean(np.std(X, axis=2) ** 2, axis=0))[0]
        self._xdawn.fit(X[:, self.picks_, :], y)
        return self

    def transform(self, X):
        X = self._reshape(X)
        return self._xdawn.transform(X[:, self.picks_, :])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class SpatialFilter(EpochsTransformerMixin):
    def __init__(self, info, estimator):
        self.info = info
        self.estimator = estimator

    def fit(self, X, y=None):
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        self.estimator.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        X = self.estimator.transform(X)
        X = np.reshape(X.T, [-1, n_epoch, n_time]).transpose([1, 0, 2])
        return X


class Reshaper(BaseEstimator, TransformerMixin):
    """Reshape data into n_samples x shape."""
    def __init__(self, shape):
        self.shape = shape

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return np.reshape(X, np.hstack((X.shape[0], self.shape)))


class LightTimeDecoding(EpochsTransformerMixin):
    def __init__(self, info, estimator=None, method='predict', n_jobs=1):
        self.info = info
        self.estimator = (LogisticRegression() if estimator is None
                          else estimator)
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = self._reshape(X)
        self.estimators_ = list()
        parallel, p_func, n_jobs = parallel_func(_fit, self.n_jobs)
        estimators = parallel(
            p_func(self.estimator, split, y)
            for split in np.array_split(X, n_jobs, axis=2))
        self.estimators_ = np.concatenate(estimators, 0)
        return self

    def transform(self, X):
        X = self._reshape(X)
        parallel, p_func, n_jobs = parallel_func(_predict_decod, self.n_jobs)
        X_splits = np.array_split(X, n_jobs, axis=2)
        est_splits = np.array_split(self.estimators_, n_jobs, axis=2)
        y_pred = parallel(
            p_func(est_split, x_split, self.method)
            for (est_split, x_split) in zip(est_splits, X_splits))

        y_pred = np.concatenate(y_pred, axis=2)
        return y_pred


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
            y_pred = _init_pred(_y_pred, X, method)
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _init_pred(y_pred, X, method):
    n_sample, n_chan, n_time = X.shape
    if method == 'predict_proba':
        n_dim = y_pred.shape[-1]
        y_pred = np.empty((n_sample, n_time, n_dim))
    else:
        y_pred = np.empty((n_sample, n_time))
    return y_pred


class LightGAT(LightTimeDecoding):
    def transform(self, X):
        X = self._reshape(X)
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
            y_pred = _init_pred_gat(_y_pred, X, len(estimators), method)
        y_pred[:, ii, ...] = _y_pred
    return y_pred


def _init_pred_gat(y_pred, X, n_train, method):
    n_sample, n_chan, n_time = X.shape
    if method == 'predict_proba':
        n_dim = y_pred.shape[-1]
        y_pred = np.empty((n_sample, n_train, n_time, n_dim))
    else:
        y_pred = np.empty((n_sample, n_train, n_time))
    return y_pred


class CustomEnsemble(TransformerMixin):
    def __init__(self, estimators, method='predict'):
        self.estimators = estimators
        self.method = method

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
