import numpy as np
from numpy.testing import assert_equal
from .. import align_signals


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
