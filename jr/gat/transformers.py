import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from mne.time_frequency import single_trial_power
from mne.filter import low_pass_filter, high_pass_filter, band_pass_filter
from mne.parallel import parallel_func

from jr.meg.time_frequency import single_trial_tfr
from pyriemann.estimation import Xdawn
from nose.tools import assert_true


class EpochsTransformerMixin(TransformerMixin, BaseEstimator):
    def __init__(self, n_chan=None):
        self.n_chan = n_chan
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

    def fit(self, X, y=None):
        # implement fit to allow debugging
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _reshape(self, X):
        # Recontruct epochs
        if (X.ndim == 3) and (self.n_chan is None):
            return X
        else:
            n_epoch = len(X)
            n_time = np.prod(X.shape[1:]) // self.n_chan
            X = np.reshape(X, [n_epoch, self.n_chan, n_time])
        return X


class TimeFreqTransformerMixin(TransformerMixin, BaseEstimator):
    def __init__(self, n_chan=None, n_freq=None):
        self.n_chan = n_chan
        self.n_freq = n_freq
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))
        assert_true(self.n_freq is None or isinstance(self.n_freq, int))

    def fit(self, X, y=None):
        # implement fit to allow debugging
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def _reshape(self, X):
        if (X.ndim == 4) and (self.n_chan is None) and (self.n_freq is None):
            return X
        else:
            # Recontruct time freq epochs
            n_epoch = len(X)
            n_time = np.prod(X.shape[1:]) // (self.n_chan * self.n_freq)
            X = np.reshape(X, [n_epoch, self.n_chan, self.n_freq, n_time])
        return np.atleast_3d(X)


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


class EpochsBaseliner(EpochsTransformerMixin):
    def __init__(self, tslice=None, mode='mean', n_chan=None):
        self.n_chan = n_chan
        self.mode = mode
        self.tslice = slice(None) if tslice is None else tslice
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))
        assert_true(self.mode in ['mean', 'logratio', 'ratio', 'zscore',
                                  'percent', 'zlogratio'])
        assert_true(isinstance(self.tslice, (slice, int)))

    def transform(self, X):
        # reshape epochs
        X = self._reshape(X)
        return baseline(X, self.mode, self.tslice)


class TimeFreqBaseliner(TimeFreqTransformerMixin):
    def __init__(self, tslice=None, mode='mean', n_chan=None, n_freq=None):
        self.n_chan = n_chan
        self.n_freq = n_freq
        self.mode = mode
        self.tslice = slice(None) if tslice is None else tslice
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))
        assert_true(self.mode in ['mean', 'logratio', 'ratio', 'zscore',
                                  'percent', 'zlogratio'])
        assert_true(isinstance(self.tslice, (slice, int)))

    def transform(self, X):
        # reshape epochs
        X = self._reshape(X)
        # Baseline
        return baseline(X, self.mode, self.tslice)


class TimeFreqDecomposer(EpochsTransformerMixin):
    def __init__(self, sfreq, frequencies, use_fft=True, n_cycles=7,
                 baseline=None, baseline_mode='ratio', times=None,
                 decim=1, n_jobs=1, zero_mean=False, verbose=None,
                 output='power', n_chan=None):
        self.frequencies = np.array(frequencies)
        self.use_fft = use_fft
        self.n_cycles = n_cycles
        self.baseline = baseline
        self.baseline_mode = baseline_mode
        self.times = times
        self.decim = decim
        self.n_jobs = n_jobs
        self.zero_mean = zero_mean
        self.verbose = verbose
        self.output = output
        self.n_chan = n_chan
        self.sfreq = sfreq
        assert_true(isinstance(sfreq, (int, float)))
        assert_true((self.frequencies.ndim == 1) and len(self.frequencies))
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

    def transform(self, X):
        # Recontruct epochs
        X = self._reshape(X)

        # Time Frequency decomposition
        kwargs = dict(sfreq=self.sfreq, frequencies=self.frequencies,
                      use_fft=self.use_fft, n_cycles=self.n_cycles,
                      decim=self.decim, n_jobs=self.n_jobs,
                      zero_mean=self.zero_mean, verbose=self.verbose)
        if self.output == 'power':
            tfr = single_trial_power(X, baseline=self.baseline,
                                     baseline_mode=self.baseline_mode,
                                     times=self.times,
                                     **kwargs)
        else:
            tfr = single_trial_tfr(X, **kwargs)
        return tfr


class TimePadder(EpochsTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, n_sample, value=0., n_chan=None):
        self.n_sample = n_sample
        assert_true(isinstance(self.n_sample, int))
        self.value = value
        assert_true(isinstance(value, (int, float)) or (value == 'median'))
        self.n_chan = n_chan
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

    def transform(self, X):
        X = self._reshape(X)
        if self.value == 'median':
            coefs = np.median(X, axis=2)
        else:
            coefs = self.value * np.ones(X.shape[:2])
        coefs = np.tile(coefs, [self.n_sample, 1, 1]).transpose([1, 2, 0])
        X = np.concatenate((coefs, X, coefs), axis=2)
        return X

    def inverse_transform(self, X):
        X = self._reshape(X)
        X = X[:, :, self.n_sample:-self.n_sample]
        return X


class TimeSelector(EpochsTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, tslice, n_chan=None):
        self.tslice = tslice
        self.n_chan = n_chan
        assert_true(isinstance(self.tslice, (slice, int)))
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        X = X[:, :, self.tslice]
        return X


class TimeFreqSelector(TimeFreqTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, tslice=None, fslice=None, n_chan=None, n_freq=None):
        self.tslice = slice(None) if tslice is None else tslice
        self.fslice = slice(None) if fslice is None else fslice
        self.n_chan = n_chan
        self.n_freq = n_freq
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))
        assert_true(self.n_freq is None or isinstance(self.n_freq, int))
        assert_true(isinstance(self.tslice, (slice, int)))
        assert_true(isinstance(self.fslice, (slice, int)))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        X = X[:, :, :, self.tslice]
        X = X[:, :, self.fslice, :]
        return X


class MyXDawn(EpochsTransformerMixin):
    """Wrapper for pyriemann Xdawn + robust.
    Will eventually need to clean both MNE and pyriemann with refactorings"""

    def __init__(self, n_filter=4, estimator='scm', n_chan=None):
        self.n_filter = n_filter
        assert_true(isinstance(self.n_filter, int))
        self.estimator = estimator
        assert_true(isinstance(estimator, str))
        self._xdawn = Xdawn(nfilter=n_filter, estimator=estimator)
        self.n_chan = n_chan
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

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
    def __init__(self, estimator, n_chan=None):
        self.n_chan = n_chan
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))
        self.estimator = estimator
        assert_true(isinstance(estimator, TransformerMixin))

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
    def __init__(self, shape=None):
        self.shape = [-1] if shape is None else shape

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return np.reshape(X, np.hstack((X.shape[0], self.shape)))


class LightTimeDecoding(EpochsTransformerMixin):
    def __init__(self, estimator=None, method='predict', n_jobs=1,
                 n_chan=None):
        self.estimator = (LogisticRegression() if estimator is None
                          else estimator)
        self.method = method
        assert_true(self.method in ['predict', 'predict_proba'])
        assert_true(hasattr(self.estimator, method))
        self.n_jobs = n_jobs
        assert_true(isinstance(self.n_jobs, int))
        self.n_chan = n_chan
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

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
        est_splits = np.array_split(self.estimators_, n_jobs)
        y_pred = parallel(
            p_func(est_split, x_split, self.method)
            for (est_split, x_split) in zip(est_splits, X_splits))

        y_pred = np.concatenate(y_pred, axis=1)
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


class GenericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, function, **fit_params):
        self.function = function
        self.fit_params = fit_params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.function(X, **self.fit_params)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)


class Filterer(EpochsTransformerMixin):
    def __init__(self, sfreq, l_freq=None, h_freq=None, filter_length='10s',
                 l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, n_jobs=1,
                 method='fft', iir_params=None, n_chan=None):
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
        self.n_chan = n_chan
        assert_true((l_freq is None) or isinstance(l_freq, (int, float)))
        assert_true((h_freq is None) or isinstance(h_freq, (int, float)))
        assert_true(self.n_chan is None or isinstance(self.n_chan, int))

    def transform(self, X, y=None):
        X = self._reshape(X)

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
