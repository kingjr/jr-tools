import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import single_trial_power
from pyriemann.estimation import Xdawn


class EpochsTransformerMixin(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        # implement fit to allow debugging
        return self

    def fit_transform(self, X, y=None):
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
