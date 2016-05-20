# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_equal
from jr.stats import fast_mannwhitneyu


def _parallel_scorer(y_true, y_pred, func, n_jobs=1):
    from nose.tools import assert_true
    from mne.parallel import parallel_func, check_n_jobs
    # check dimensionality
    assert_true(y_true.ndim == 1)
    assert_true(y_pred.ndim == 2)
    # set jobs not > to n_chunk
    n_jobs = min(y_pred.shape[1], check_n_jobs(n_jobs))
    parallel, p_func, n_jobs = parallel_func(func, n_jobs)
    chunks = np.array_split(y_pred.transpose(), n_jobs)
    # run parallel
    out = parallel(p_func(chunk.T, y_true) for chunk in chunks)
    # gather data
    return np.concatenate(out, axis=0)


def _check_y(y_true, y_pred):
    """Aux function to apply scorer across multiple dimensions."""
    # Reshape to get 2D
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert_true(len(y_pred) == len(y_pred))
    shape = y_pred.shape
    y_pred = np.reshape(y_pred, [shape[0], -1])
    y_true = np.squeeze(y_true)
    assert_true(y_true.ndim == 1)
    # remove nan values XXX non-adjacency need memory!
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        sel = np.where(~np.isnan(y_true[:, np.newaxis] + y_pred))[0]
        y_true = y_true[sel]
        y_pred = y_pred[sel, :]
    return y_true, y_pred, shape


def scorer_spearman(y_true, y_pred, n_jobs=1):
    from jr.stats import repeated_spearman
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    rho = _parallel_scorer(y_true, y_pred, repeated_spearman, n_jobs)
    if (len(shape) > 1) and (np.sum(shape[1:]) > 1):
        rho = np.reshape(rho, shape[1:])
    else:
        rho = rho[0]
    return rho


def scorer_corr(y_true, y_pred, n_jobs=1):
    from jr.stats import repeated_corr
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    rho = _parallel_scorer(y_true, y_pred, repeated_corr, n_jobs)
    if (len(shape) > 1) and (np.sum(shape[1:]) > 1):
        rho = np.reshape(rho, shape[1:])
    else:
        rho = rho[0]
    return rho


def scorer_auc(y_true, y_pred, n_jobs=1):
    from sklearn.preprocessing import LabelBinarizer
    shape = y_pred.shape
    if len(shape) == 1:
        y_pred = y_pred[:, np.newaxis]
    le = LabelBinarizer()
    y_true = le.fit_transform(y_true)
    assert_array_equal(np.unique(y_true), [0, 1])
    _, _, auc = fast_mannwhitneyu(y_pred[y_true == 0, ...],
                                  y_pred[y_true == 1, ...], n_jobs=n_jobs)
    return auc


def prob_accuracy(y_true, y_pred, **kwargs):
    from sklearn.metrics import accuracy_score
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred, **kwargs)


def scorer_angle(y_true, y_pred, n_jobs=1):
    """Scoring function dedicated to AngularRegressor"""
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    accuracy = _parallel_scorer(y_true, y_pred, _angle_accuracy, n_jobs)
    if (len(shape) > 1) and (np.sum(shape[1:]) > 1):
        accuracy = np.reshape(accuracy, shape[1:])
    else:
        accuracy = accuracy[0]
    return accuracy


def _angle_accuracy(y_pred, y_true):  # XXX note reversal of y_true & y_pred
    angle_error = y_true[:, np.newaxis] - y_pred
    score = np.mean(np.abs((angle_error + np.pi) % (2 * np.pi) - np.pi),
                    axis=0)
    return np.pi / 2 - score


def scorer_circLinear(y_line, y_circ):
    R, R2, pval = circular_linear_correlation(y_line, y_circ)
    return R
