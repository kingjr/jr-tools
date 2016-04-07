# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def scorer_spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    sel = np.where(~np.isnan(y_true + y_pred))[0]
    rho, p = spearmanr(y_true[sel], y_pred[sel])
    return rho


def scorer_corr(y_true, y_pred):
    from scipy.stats import pearsonr
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    sel = np.where(~np.isnan(y_true + y_pred))[0]
    rho, p = pearsonr(y_true[sel], y_pred[sel])
    return rho


def scorer_auc(y_true, y_pred):
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


def scorer_angle(truth, prediction):
    """Scoring function dedicated to AngularRegressor"""
    angle_error = truth - prediction[:, 0]
    pi = np.pi
    score = np.mean(np.abs((angle_error + pi) % (2 * pi) - pi))
    return np.pi / 2 - score


def scorer_circLinear(y_line, y_circ):
    R, R2, pval = circular_linear_correlation(y_line, y_circ)
    return R
