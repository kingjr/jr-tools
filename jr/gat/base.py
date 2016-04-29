from copy import deepcopy
import numpy as np
from mne.decoding import GeneralizationAcrossTime
from ..meg import make_meta_epochs


def get_diagonal_ypred(gat):
    return _get_diagonal(gat.y_pred_,
                         gat.train_times_['times'],
                         gat.test_times_['times'],
                         gat.train_times_['step'])


def get_diagonal_score(gat):
    return _get_diagonal(gat.score,
                         gat.train_times_['times'],
                         gat.test_times_['times'],
                         gat.train_times_['step'])


def _get_diagonal(y_pred, train_times, test_times, step):
    diag = list()
    n_trials, n_dim = y_pred[0][0].shape
    for train_idx, train_time in enumerate(train_times):
        # find closest testing time from train_time
        lag = test_times[train_idx] - train_time
        test_idx = np.abs(lag).argmin()
        # check that not more than 1 classifier away
        if np.abs(lag[test_idx]) > step:
            diag.append(np.nan * np.ones((n_trials, n_dim)))
        else:
            diag.append(y_pred[train_idx][test_idx])
    return diag


def update_pred(gat, y_pred, copy=True):
    if copy:
        gat = deepcopy(gat)
    gat.y_pred_ = y_pred
    if hasattr(gat, 'testing_times_'):
        times = gat.test_times_['times']
        slices = gat.test_times_['slices']
        for pred, time, slic in zip(y_pred, times, slices):
            n = len(pred)
            sfreq = len(time) / np.ptp(time)
            gat.test_times_['times'] = np.cumsum(np.ones(n)) / sfreq + time[0]
            gat.test_times_['slices'] = np.arange(n) + slic[0]
    else:
        time = gat.train_times_['times']
        slic = gat.train_times_['slices']
        sfreq = len(time) / np.ptp(time)
        for pred in y_pred:
            n = len(pred)
            gat.test_times_['times'] = \
                [np.cumsum(np.ones(n)) / sfreq + time[0]] * len(time)
            gat.test_times_['slices'] = [np.arange(n) + slic[0]] * len(time)
    if copy:
        return gat


def equalize_samples(y):
    count = list()
    out = list()
    for ii in np.unique(y):
        count.append(len(np.where(y == ii)[0]))
    for ii in np.unique(y):
        sel = np.where(y == ii)[0]
        np.random.shuffle(sel)
        out = np.hstack((out, sel[:np.min(count)]))
    return np.array(out, dtype=int)


def subscore(gat, sel, y=None, scorer=None):
    """Subscores a GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        sel : list or array, shape (n_predictions)
        y : None | list or array, shape (n_selected_predictions,)
            If None, y set to gat.y_true_. Defaults to None.

    Returns
    -------
    scores
    """
    gat_ = subselect_ypred(gat, sel)
    if scorer is not None:
        gat_.scorer = scorer
    return gat_.score(y=y)


def subselect_ypred(gat, sel):
    """Select subselection of y_pred_ of GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        sel : list or array, shape (n_predictions)

    Returns
    -------
        new gat
    """
    gat_ = deepcopy(gat)
    try:
        gat_.y_pred_ = np.array(gat_.y_pred_)
        gat_.y_pred_ = gat_.y_pred_[:, :, sel, :]
    except TypeError:
        # Subselection of trials
        for train in range(len(gat_.y_pred_)):
            for test in range(len(gat_.y_pred_[train])):
                gat_.y_pred_[train][test] = gat_.y_pred_[train][test][sel, :]
    try:
        gat_.y_train_ = gat_.y_train_[sel]
    except:
        gat_.y_train_ = None
    return gat_


def mean_ypred(gat, y=None, classes=None, sel=None):
    """Provides mean prediction for each category.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        y : None | list or array, shape (n_predictions,)
            If None, y set to gat.y_train_. Defaults to None.
        classes : int | list of int
            The classes to be averaged. Defaults to np.unique(y).
            If [c not in y for c in classes], returns np.nan
        sel : list of indices
            Selected y_pred. Defaults to range(n_predictions).
            /!\ If y given, expect to be able to do y[sel].

    Returns
    -------
    mean_y_pred : list of list of (float | array) | np.ndarray
                  shape (n_train_time, n_test_time, n_classes, n_predict_shape)
        The mean prediction for each training and each testing time point
        for each class.
    """
    if sel is None:
        sel = range(len(gat.y_pred_[0][0]))
    if y is None:
        y = gat.y_train_[sel]
    if classes is None:
        classes = np.unique(y)
    try:
        gat.y_pred_ = np.array(gat.y_pred_)
        nT, nt, _, ndim = gat.y_pred_.shape
        y_pred = np.zeros((nT, nt, len(classes), ndim))
        for ii, c in enumerate(classes):
            sel = y == c
            if sum(sel):
                y_pred[:, :, ii, :] = np.mean(gat.y_pred_[:, :, sel, :],
                                              axis=2)
            else:
                y_pred[:, :, ii, :] = np.nan

    except ValueError:
        y_pred = list()
        for train in range(len(gat.y_pred_)):
            y_pred_ = list()
            for test in range(len(gat.y_pred_[train])):
                y_pred__ = list()
                for c in classes:
                    sel = y == c
                    if sum(sel):
                        y_pred = gat.y_pred_[train][test][sel, :]
                        m = np.mean(y_pred[y == c, :], axis=0)
                    else:
                        m = np.nan
                    y_pred__.append(m)
                y_pred_.append(y_pred__)
            y_pred.append(y_pred_)
    return y_pred


def rescale_ypred(gat, clf=None, scorer=None, keep_sign=True):
    """"""
    if clf is None:
        clf = gat.clf
    if scorer is None:
        scorer = gat.scorer_

    y_pred_r = np.empty_like(gat.y_pred_)
    for t_train, y_pred_ in enumerate(gat.y_pred_):
        for t_test, y_pred__ in enumerate(y_pred_):
            for train, test in gat._cv_splits:
                n = len(y_pred__)
                X = np.reshape(y_pred__[:, 0], [n, 1])
                clf.fit(X[train], gat.y_train_[train])
                proba = clf.predict_proba(X[test])
                if keep_sign:
                    if scorer(gat.y_train_[train],
                              y_pred__[train].squeeze()) < .5:
                        proba[test, 0] = -proba[test, 0] + 1
                y_pred_r[t_train][t_test][test] = proba
    return y_pred_r


def zscore_ypred(gat):
    """"""
    y_pred = deepcopy(gat.y_pred_)
    n_T = len(gat.train_times_['slices'])
    for t_train in range(n_T):
        n_t = len(gat.test_times_['slices'][t_train])
        for t_test in range(n_t):
            p = y_pred[t_train][t_test]
            p -= np.tile(np.mean(p, axis=0), [len(p), 1])
            p /= np.tile(np.std(p, axis=0), [len(p), 1])
            y_pred[t_train][t_test] = p
    return y_pred


class GAT(GeneralizationAcrossTime):
    def __init__(self, gat):
        for key in gat.__dict__.keys():
            setattr(self, key, getattr(gat, key))

    def score(self, y=None, scorer=None):
        if scorer is not None:
            self.scorer = scorer
        return super(GAT, self).score(y=y)

    def subscore(self, sel, y=None, scorer=None, copy=True):
        """Subscores a GAT.

        Parameters
        ----------
            sel : list or array, shape (n_predictions)
            y : None | list or array, shape (n_selected_predictions,)
                If None, y set to gat.y_true_. Defaults to None.
            copy : bool
                change GAT object

        Returns
        -------
        scores
        """
        scores = subscore(self)
        if copy is True:
            self.scores_ = scores
        return scores

    def subselect_ypred(self, sel, copy=False):
        """Select subselection of y_pred_ of GAT.

        Parameters
        ----------
            sel : list or array, shape (n_predictions)
            copy : bool
                change GAT object

        Returns
        -------
            new gat
        """
        return subselect_ypred(self, sel)

    def mean_ypred(self, y=None):
        """Provides mean prediction for each category.

        Parameters
        ----------
            y : None | list or array, shape (n_predictions,)
                If None, y set to gat.y_train_. Defaults to None.

        Returns
        -------
        mean_y_pred : list of list of (float | array),
                      shape (train_time, test_time, classes, predict_shape)
            The mean prediction for each training and each testing time point
            for each class.
        """
        return mean_ypred(self, y=y)

    def rescale_ypred(self, clf=None, scorer=None, keep_sign=True):
        """"""
        return rescale_ypred(self, clf=clf, scorer=scorer, keep_sign=keep_sign)

    def zscore_ypred(self):
        """"""
        return zscore_ypred(self)


class GATs(object):
    def __init__(self, gat_list, remove_coef=True):
        if isinstance(gat_list, GeneralizationAcrossTime):
            gat_list = [gat_list]

        if remove_coef:
            gat_list_ = list()
            for gat in gat_list:
                gat.estimators_ = list()
                gat_list_.append(gat)
            gat_list = gat_list_

        self.gat_list = gat_list
        for key in gat_list[0].__dict__().key():
            setattr(self, key, getattr(gat_list[0], key))
        if hasattr(gat_list[0], 'y_pred_'):
            self._mean_y_pred_()
        if hasattr(gat_list[0], 'scores_'):
            self._mean_scores()
        return self

    def _mean_ypreds(self):
        y_pred_list = list()
        for gat in self.gat_list:
            y_pred_list.append(gat.mean_ypred())
        self.y_pred_ = deepcopy(y_pred_list[0])
        for train, y_pred_train in enumerate(self.y_pred):
            for test in range(len(y_pred_train)):
                self.y_pred_[train][test] = np.mean(y_pred_list[train][test],
                                                    axis=0)
        self.y_train_ = np.unique([gat.y_train_ for gat in self.gat_list])

    def _mean_scores(self):
        self.scores_ = np.mean([gat.scores_ for gat in self.gat_list], axis=0)

    def mean_ypreds(self, y=None):
        return self.y_preds_

    def score(self, y_list=None):
        if np.ndim(y_list) < len(self.gat_list):
            y_list = [y_list for idx in range(len(self.gat_list))]

        for gat, y in zip(self.gat_list, y_list):
            gat.score(y=y)

        self.scorer = gat.scorer_
        self.gat.scores_ = np.mean([gat.scores_ for gat in self.gat_list],
                                   axis=0)
        return self.gat.scores_


def combine_y(gat_list, order=None, n_pred=None):
    """Combines multiple gat.y_pred_ & gat.y_train_ into a single gat.

    Parameters
    ----------
        gat_list : list of GeneralizationAcrossTime objects, shape (n_gat)
            The gats must have been predicted (gat.predict(epochs))
        order : None | list, shape (n_gat), optional
            Order of the prediction, to be recombined. Defaults to None.
        n_pred : None | int, optional
            Maximum number of predictions. If None, set to max(sel). Defaults
            to None.
    Returns
    -------
        cmb_gat : GeneralizationAcrossTime object
            The combined gat object"""
    from gat.utils import GAT

    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]
        order = [order]

    for gat in gat_list:
        if not isinstance(gat, GeneralizationAcrossTime):
            raise ValueError('gat must be a GeneralizationAcrossTime object')
    gat_list = [GAT(gat) for gat in gat_list]

    if order is not None:
        if len(gat_list) != len(order):
            raise ValueError('len(order) must equal len(gat_list)')
    else:
        order = [range(len(gat.y_pred_[0][0])) for gat in gat_list]
        for idx in range(1, len(order)):
            order[idx] += len(order[idx-1])

    # Identifiy trial number
    if n_pred is None:
        n_pred = np.max([np.max(sel) for sel in order]) + 1
    n_dims = np.shape(gat_list[0].y_pred_[0][0])[1]

    # Initialize combined gat
    cmb_gat = deepcopy(gat_list[0])

    # Initialize y_pred
    cmb_gat.y_pred_ = list()
    cmb_gat.cv_.n = n_pred
    cmb_gat.cv_.test_folds = np.nan * np.zeros(n_pred)
    cmb_gat.cv_.y = np.nan * np.zeros(n_pred)

    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred_.append(np.nan * np.ones((n_pred, n_dims)))
        cmb_gat.y_pred_.append(y_pred_)

    # Initialize y_train
    cmb_gat.y_train_ = np.ones((n_pred,))

    for gat, sel in zip(gat_list, order):
        cmb_gat.y_train_[sel] = gat.y_train_
        cmb_gat.cv_.test_folds[sel] = gat.cv_.test_folds
        cmb_gat.cv_.y[sel] = gat.cv_.y
        for t_train in range(len(gat.y_pred_)):
            for t_test in range(len(gat.y_pred_[t_train])):
                cmb_gat.y_pred_[t_train][t_test][sel, :] = \
                    gat.y_pred_[t_train][t_test]
    # clean
    for att in ['scores_', 'scorer_', 'y_true_']:
        if hasattr(cmb_gat, att):
            delattr(cmb_gat, att)
    return cmb_gat


class MetaGAT(object):
    def __init__(self, gat, n=100):
        self.cv = gat.cv
        self.gat = gat
        self.n = n

    def fit(self, epochs, y=None):
        from sklearn.cross_validation import check_cv, StratifiedKFold
        from mne.decoding.time_gen import _check_epochs_input
        X, y, self.gat.picks_ = _check_epochs_input(epochs, y, self.gat.picks)
        gat_list = list()

        cv = self.cv
        if isinstance(cv, (int, np.int)):
            cv = StratifiedKFold(y, cv)
        cv = check_cv(cv, X, y, classifier=True)
        # Construct meta epoch and fit gat with a single fold
        for ii, (train, test) in enumerate(cv):
            # meta trial
            epochs_ = make_meta_epochs(epochs[train], y[train], n_bin=self.n)
            # fit gat
            gat_ = deepcopy(self.gat)
            cv_one_fold = [(range(len(epochs_)), [])]
            gat_.cv = cv_one_fold
            gat_.fit(epochs_, epochs_.events[:, 2])
            gat_list.append(gat_)

        # gather
        self.gat = gat_
        self.gat.train_times_ = gat_.train_times_
        self.gat.estimators_ = np.squeeze(
            [gat.estimators_ for gat in gat_list]).T.tolist()
        self.gat.cv_ = cv
        self.gat.y_train_ = y

    def predict(self, epochs):
        return self.gat.predict(epochs)

    def score(self, epochs=None, y=None):
        return self.gat.score(epochs, y)


class SensorDecoding():
    """Fit an estimator on each sensor separately across all time points"""
    def __init__(self, clf=None, predict_method='predict', scorer=None,
                 n_jobs=1, ch_groups=None, cv=5):
        self.clf = clf
        self.predict_method = predict_method
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.ch_groups = ch_groups
        self.cv = cv

    def fit(self, epochs, y=None):
        from mne.decoding import TimeDecoding
        epochs = self._prepare_data(epochs)
        self._td = TimeDecoding(clf=self.clf, cv=self.cv,
                                predict_method=self.predict_method,
                                scorer=self.scorer, n_jobs=self.n_jobs,)
        self._td.fit(epochs, y=y)

    def predict(self, epochs):
        epochs = self._prepare_data(epochs)
        self._td.predict(epochs)
        self.y_pred_ = self._td.y_pred_
        return self.y_pred_

    def score(self, epochs=None, y=None):
        if not hasattr(self, 'y_pred_'):
            self.predict(epochs)
        self._td.score(y=y)
        self.scores_ = self._td.scores_
        return self.scores_

    def _prepare_data(self, epochs):
        """Swap channel and time"""
        from mne import EpochsArray, create_info
        # regroup channels
        X = self._group_channels(epochs)
        # swap time and channels
        X = X.transpose([0, 2, 1])
        # format for GAT
        epochs_ = EpochsArray(
            data=X,
            events=epochs.events,
            info=create_info(X.shape[1], sfreq=1, ch_types='mag'))
        return epochs_

    def _group_channels(self, epochs):
        if self.ch_groups is None:
            self.ch_groups = np.arange(len(epochs.ch_names))[None, :]
        if self.ch_groups.ndim != 2:
            raise ValueError('Channel groups must be n_group * n_chans array')
        n_group = len(self.ch_groups)
        n_time = len(epochs.times)
        X = np.empty((len(epochs), n_group, self.ch_groups.shape[1] * n_time))
        for ii, chans in enumerate(self.ch_groups):
            X[:, ii, :] = np.hstack([epochs._data[:, ch, :] for ch in chans])
        return X


class TimeFrequencyDecoding():
    """Search light across sensor in each time-frequency bin."""
    def __init__(self,  freqs, tfr_kwargs=None, td=None, n_jobs=1):
        from mne.decoding import TimeDecoding
        # Search light parameters
        self.td = TimeDecoding() if td is None else td
        self.td.n_jobs = n_jobs
        if not isinstance(self.td, TimeDecoding):
            raise ValueError('`td` needs to be a `TimeDecoding` object, got '
                             '%s instead.' % type(td))
        if (('step' in self.td.times.keys()) or
                ('length' in self.td.times.keys())):
            raise ValueError("Cannot use advance `time` param")

        # Time frequency decomposition parameters
        self.tfr_kwargs = tfr_kwargs
        if tfr_kwargs is None:
            self.tfr_kwargs = dict()
        self.tfr_kwargs['n_jobs'] = n_jobs
        self.tfr_kwargs['frequencies'] = freqs
        self.freqs = freqs

    def transform(self, epochs):
        from mne import EpochsArray
        from mne.time_frequency import single_trial_power
        # Time Frequency decomposition
        tfr = single_trial_power(epochs._data, sfreq=epochs.info['sfreq'],
                                 **self.tfr_kwargs)

        # Consider frequencies as if it was different time points
        n_trial, n_chan, n_freq, n_time = tfr.shape
        tfr = np.reshape(tfr, [n_trial, n_chan, n_freq * n_time])

        # Make pseudo epochs
        sfreq = epochs.info['sfreq']
        if 'decim' in self.tfr_kwargs.keys():
            sfreq /= self.tfr_kwargs['decim']
        info = epochs.info.copy()
        info['sfreq'] = sfreq
        self._tfr_epochs = EpochsArray(data=tfr, info=info,
                                       events=epochs.events)

    def fit(self, epochs=None, y=None):
        self._check_transform(epochs)
        self.td.fit(self._tfr_epochs, y=y)
        return self

    def predict(self, epochs=None):
        self._check_transform(epochs)
        self.td.predict(self._tfr_epochs)

    def y_pred_(self):
        nT, nt, ns, np = self.td.y_pred_.shape
        nfreq = len(self.tfr_kwargs['frequencies'])
        return np.reshape(self.td.y_pred_, [nfreq, nT, ns, np])

    def score(self, epochs=None, y=None):
        if epochs is not None:
            self._check_transform(epochs)
        epochs = self._tfr_epochs
        scores = self.td.score(epochs, y=y)
        self.scores_ = np.reshape(scores, [len(self.freqs), -1])
        return self.scores_

    def _check_transform(self, epochs):
        if epochs is not None:
            self.transform(epochs)
        if not hasattr(self, '_tfr_epochs'):
            raise RuntimeError('You need to transform epochs first')
