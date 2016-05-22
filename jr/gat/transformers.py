import numpy as np
from sklearn.base import TransformerMixin


class EpochsTransformerMixin(TransformerMixin):

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


class Baseliner(EpochsTransformerMixin):
    def __init__(self, info, scaler=None, tslice=None):
        from sklearn.preprocessing import StandardScaler
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
        from mne.time_frequency import single_trial_power
        from nose.tools import assert_true
        import warnings
        sfreq = self.info['sfreq']

        # Recontruct epochs
        X = self._reshape(X)

        # Check whether wavelets are too long
        min_freq = np.min(self.frequencies)
        min_wavelet = 1. / min_freq * self.n_cycles
        n_pad_sec = min_wavelet - (X.shape[2] / sfreq)
        if n_pad_sec > 0:
            warnings.warn('Epoch too short! Padding before time freq...')
            n_pad = n_pad_sec * sfreq // 2
            bsl = np.median(X, axis=2)
            bsl = np.tile(bsl, [n_pad, 1, 1]).transpose([1, 2, 0])
            X = np.concatenate((bsl, X, bsl), axis=2)

        # Time Frequency decomposition
        tfr = single_trial_power(
            X, sfreq=sfreq, frequencies=self.frequencies, use_fft=self.use_fft,
            n_cycles=self.n_cycles, baseline=self.baseline,
            baseline_mode=self.baseline_mode, times=self.times,
            decim=self.decim, n_jobs=self.n_jobs, zero_mean=self.zero_mean,
            verbose=self.verbose)

        # Remove padding
        if n_pad_sec > 0:
            tfr = tfr[:, :, :, n_pad:-n_pad]
            assert_true(tfr.shape[-1] == X.shape[2])

        return tfr


class TimePadder(EpochsTransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, info, n_sample):
        self.n_sample = n_sample
        self.info = info

    def transform(self, X):
        X = self._reshape(X)
        coefs = np.median(X, axis=2)
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


class TimeFreqCropper(TransformerMixin):
    """Padd time before and after epochs"""
    def __init__(self, info, n_sample, frequencies):
        self.n_sample = n_sample
        self.info = info
        self.frequencies = frequencies

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X, self)
        X = X[:, :, self.n_sample:-self.n_sample]
        return X.reshape([len(X), -1])

    def _reshape(self, X):
        # Recontruct epochs
        n_epoch = len(X)
        n_chan = len(self.info['chs'])
        n_time = np.prod(X.shape[1:]) // n_chan
        n_freq = len(self.frequencies)
        X = np.reshape(X, [n_epoch, n_chan, n_freq, n_time])
        return X


class MyXDawn(EpochsTransformerMixin):
    """Wrapper for pyriemann Xdawn + robust.
    Will eventually need to clean both MNE and pyriemann with refactorings"""

    def __init__(self, info, n_filter=4, estimator='scm'):
        from pyriemann.estimation import Xdawn
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
