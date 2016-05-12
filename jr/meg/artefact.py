# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence : GNU GPLv3

import numpy as np
from sklearn.base import BaseEstimator
from mne.parallel import parallel_func


class SelfRegression(BaseEstimator):
    """ Fit a series of regressors that aim at predicting each feature when
    the latter is hidden from the regressors.

    Parameters
    ----------
    estimator : sklearn regressor | None
        The regressor. Defaults to LinearRegression()
    n_jobs : int
        The number of parallel cores.

    Attributes
    ----------
    estimators_ : array, shape (n_feature)
        The array of fitted estimator for each feature.
    y_pred_ : array, shape(n_samples, n_feature)
        The predictions.
    """
    def __init__(self, estimator=None, n_jobs=-1):
        from mne.parallel import check_n_jobs
        from sklearn.linear_model import LinearRegression
        self.estimator = LinearRegression() if estimator is None else estimator
        self.n_jobs = n_jobs = check_n_jobs(n_jobs)

    def fit(self, X):
        """Fits a regressor for each feature.

        Parameters
        ----------
        X : array, shape (n_sample, n_feature)
            The data.
        """
        from sklearn.base import clone
        n_sample, self.n_feature_ = X.shape
        # Setup parallel
        n_splits = n_jobs = np.min([self.n_jobs, self.n_feature_])
        parallel, p_func, n_jobs = parallel_func(_fit_loop, n_jobs,
                                                 verbose=None,
                                                 max_nbytes='auto')
        # Split chunks of features to avoid overheads
        splits = np.array_split(np.arange(self.n_feature_), n_splits)
        out = parallel(p_func([clone(self.estimator) for f in split], X, split)
                       for split in splits)
        self.estimators_ = np.concatenate(out, axis=0)

    def predict(self, X):
        """Predict all features.

        Parameters
        ----------
        X : array, shape (n_sample, n_feature)
            The data.

        Returns
        -------
        X_pred : array, shape(n_sample, n_feature)
        """
        n_sample, n_feature = X.shape
        if n_feature != self.n_feature_:
            raise ValueError('X must have same dims in fit and predict.')
        n_splits = n_jobs = np.min([self.n_jobs, self.n_feature_])
        parallel, p_func, n_jobs = parallel_func(_predict_loop, n_jobs,
                                                 verbose=None,
                                                 max_nbytes='auto')

        splits = np.array_split(np.arange(n_feature), n_splits)
        y_pred = parallel(p_func(self.estimators_[split], X, split)
                          for split in splits)
        self.y_pred_ = np.hstack(y_pred)
        return self.y_pred_


def _fit_loop(estimators, X, split):
    """Auxiliary functions of SelfRegression"""
    _, n_feature = X.shape
    for feature, estimator in zip(split, estimators):
        features = np.delete(np.arange(n_feature), feature)
        estimator.fit(X[:, features], y=X[:, feature])
    return estimators


def _predict_loop(estimators, X, split):
    """Auxiliary functions of SelfRegression"""
    n_sample, n_feature = X.shape
    y_pred = np.zeros((n_sample, len(split)))
    for f_idx, (feature, estimator) in enumerate(zip(split, estimators)):
        features = np.delete(np.arange(n_feature), feature)
        y_pred[:, f_idx] = estimator.predict(X[:, features])
    return y_pred


def detect_bad_channels(raw, estimator=None, n_train=1e4, n_test=1e4,
                        n_jobs=-1, picks=None,):
    """This example shows how EEG/MEG bad channel detection can be done by
    trying to predict the value of each channel of each time point from the
    activity of all other channels at the corresponding time points.

    Indeed, knowning the high spatial correlation of EEG/MEG signals, a given
    channel can be considered as noisy if it doesn't (anti)correlate with any
    other channels.

    Note that:
    - this this doesn't work for intracranial EEG, where the spatial
    correlation is much smaller.
    - this method isn't ideal to identify bad timing. For this, I would
    recommend Alex. Barachant's Potato algorithm available at
    http://github.com/abarachant/pyRiemann

    """
    from mne import pick_types
    from sklearn.preprocessing import RobustScaler
    # Subsample times for faster computation
    # Note that, considering that n_sample >> n_feature, a real cross-
    # validation isn't really necessary
    times = np.arange(len(raw.times))
    np.random.shuffle(times)
    times = times[:(n_train + n_test)]

    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True)
    X = raw._data[picks, :][:, times].T.copy()

    # To be consistent across chan types, we'll normalize the data:
    X = RobustScaler().fit_transform(X)
    n_time, n_chan = X.shape

    # Fit
    art = SelfRegression(estimator=estimator, n_jobs=n_jobs)
    art.fit(X[:n_train, :])
    Xpred = art.predict(X[-n_test:, :])

    # Score
    errors = (Xpred-X[-n_test:, :]) ** 2

    return errors


def remove_linenoise(raw, noise_freq, width=2, shuffle_time=True, decim=100,
                     n_component=1, plot=False, copy=True, picks=None,
                     harmonics=True):
    import matplotlib.pyplot as plt
    from mne import pick_types
    from mne.preprocessing import ICA
    from mne.time_frequency.psd import psd_welch

    # Setup line frequency
    if isinstance(noise_freq, str):
        # automatic harmonics
        if noise_freq == 'us':
            noise_freq = 60
        else:
            noise_freq = 50
    elif not isinstance(noise_freq, (float, int)):
        raise NotImplementedError('Multiple bands')

    def plot_psd(psd, freqs, ax, title):
        for psd_ in psd:
            ax.plot(freqs, np.log10(psd_))
        ax.set_xlabel('Frequencies')
        ax.set_title(title)

    if copy:
        raw = raw.copy()

    if picks is None:
        picks = pick_types(raw.info, eeg=True, meg=True, seeg=True)

    if plot:
        fig, axes = plt.subplots(1, 3, sharex=True)
        psd, freqs = psd_welch(raw, picks=picks)
        plot_psd(psd, freqs, axes[0], 'Raw Sensors')

    # Fit ICA on filtered data
    raw_ = raw.copy()
    if harmonics:
        # set up harmonics
        n_harm = raw_.info['sfreq'] // (2. * noise_freq) + 1
        harmonics = noise_freq * np.arange(1, n_harm)
        # Band pass filtering outside lowest harmonics and nquist
        raw_.filter(noise_freq - width, harmonics[-1] + width)
        # Band stop filter in between harmonics
        raw_.notch_filter(freqs=harmonics[:-1]+noise_freq//2,
                          notch_widths=noise_freq - 2*width)
    else:
        raw_.filter(noise_freq-width, noise_freq+width)

    # Shuffle time axis to avoid decimation aliasing
    if shuffle_time:
        time = np.arange(raw_.n_times)
        np.random.shuffle(time)
        raw_._data[:, time] = raw_._data
    ica = ICA(verbose=False)
    ica.fit(raw_, decim=decim, picks=picks)

    # Compute PSD of components
    raw_._data[picks, :] = np.dot(ica.mixing_matrix_, raw._data[picks, :])
    psd, freqs = psd_welch(raw_, picks=picks)
    if plot:
        plot_psd(psd, freqs, axes[1], 'Components')

    # Find noise component and remove
    freq = np.where(freqs >= noise_freq)[0][0]
    sel = np.argsort(psd[:, freq])[-n_component:].tolist()
    raw_ = ica.apply(raw, exclude=sel, copy=True)

    if plot:
        psd, freqs = psd_welch(raw_, picks=picks)
        plot_psd(psd, freqs, axes[2], 'Clean sensors')

    return raw_


def find_reference(raw, n_cluster, pick_types=None, copy=True,
                   flat_threshold=1e-15, n_split=100, plot=True):
    """ Computes covariance on splits of the raw data, and apply KMeans
    clustering to find the number of disjoint references.
    n_cluster is found with PCA if float
    """
    import matplotlib.pyplot as plt
    from pyriemann.estimation import Covariances
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances

    if copy:
        raw = raw.copy()
    # Remove flat lines
    flat = np.where(np.std(raw._data, axis=1) < flat_threshold)[0]
    for ch in flat:
        raw.info['bads'] += [raw.ch_names[ch]]

    # Pick data channels only
    if pick_types is None:
        pick_types = dict(seeg=True, exclude='bads')
    raw.pick_types(**pick_types)

    # Compute covariance on data splits
    n_time = len(raw.times)
    t_max = raw.times[n_time - n_time % n_split - 1]
    raw.crop(0, t_max, copy=False)  # ensure regularly sized splits
    X = np.array(np.array_split(raw._data, n_split, axis=1))
    covs = Covariances().fit_transform(X)

    # Compute cluster for each data split
    cluster = KMeans(n_cluster)
    all_kmeans = list()
    for cov in covs:
        dist = pairwise_distances(cov)
        all_kmeans.append(cluster.fit_predict(dist))

    # Combine clusters
    dist = pairwise_distances(np.array(all_kmeans).T)
    idx = cluster.fit_predict(dist)

    if plot:
        idx_ = np.argsort(idx)
        cov = np.median(covs, axis=0)
        plt.matshow(np.log10(cov)[idx_, :][:, idx_])

    clusters = [np.array(raw.ch_names)[idx == ii] for ii in np.unique(idx)]
    return clusters
