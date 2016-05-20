import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from .. import align_signals
from .. import fast_mannwhitneyu


def test_align_signal():
    a = np.asarray(np.random.rand(1000) > .9, float)
    for pad in [10, 11, 0]:
        b = np.hstack((np.zeros(pad), a))
        for a_, b_, sign in ((a, b, 1), (b, a, -1)):
            # default
            assert_equal(align_signals(a_, b_), sign * pad)
            # even / odd lengths
            assert_equal(align_signals(a_[:-1], b_[:-1]), sign * pad)
            assert_equal(align_signals(a_[:-1], b_), sign * pad)
            assert_equal(align_signals(a_, b_[:-1]), sign * pad)


def test_auc():
    from sklearn.metrics import roc_auc_score
    for ii in range(10):
        X = np.random.rand(20, 50)
        y = np.random.randint(0, 2, 20)
        _, _, auc = fast_mannwhitneyu(X[y == 0, ...],
                                      X[y == 1, ...])
        _, _, auc2 = fast_mannwhitneyu(X[y == 1, ...],
                                       X[y == 0, ...])
        auc3 = [roc_auc_score(y, x) for x in X.T]
        assert_array_almost_equal(auc, 1. - auc2)
        assert_array_almost_equal(auc, auc3)
