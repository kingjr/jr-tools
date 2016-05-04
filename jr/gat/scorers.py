# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

from nose.tools import assert_true
import numpy as np


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


def scorer_spearman(y_true, y_pred):
    from jr.stats import repeated_spearman
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    rho = repeated_spearman(y_pred, y_true)
    return np.reshape(rho, shape[1:])


def scorer_corr(y_true, y_pred):
    from jr.stats import repeated_corr
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    rho = repeated_corr(y_pred, y_true)
    return np.reshape(rho, shape[1:])


def scorer_auc(y_true, y_pred):
    """Only accepts 2 class 1 dim"""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    if np.ndim(y_pred) == 2:
        y_pred = np.ravel(y_pred[:, 0])
    le = LabelBinarizer()
    y_true = le.fit_transform(y_true)
    return roc_auc_score(y_true, y_pred)


def prob_accuracy(y_true, y_pred, **kwargs):
    from sklearn.metrics import accuracy_score
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred, **kwargs)


def scorer_angle(y_true, y_pred):
    """Scoring function dedicated to AngularRegressor"""
    y_true, y_pred, shape = _check_y(y_true, y_pred)
    angle_error = y_true[:, np.newaxis] - y_pred
    score = np.mean(np.abs((angle_error + np.pi) % (2 * np.pi) - np.pi),
                    axis=0)
    accuracy = np.pi / 2 - score
    return np.reshape(accuracy, shape[1:])


def scorer_circLinear(y_line, y_circ):
    R, R2, pval = circular_linear_correlation(y_line, y_circ)
    return R
