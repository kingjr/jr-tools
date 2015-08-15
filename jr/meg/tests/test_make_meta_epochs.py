import numpy as np
from nose.tools import assert_true, assert_equal
from ..base import mat2mne, make_meta_epochs


def test_make_meta_epochs():
    ntrial, nchan, ntime = 1000, 32, 10
    epochs = mat2mne(np.random.randn(ntrial, nchan, ntime))
    y = np.arange(ntrial) * 1e3
    mepochs = make_meta_epochs(epochs, y, n_bin=100)
    assert_equal(len(epochs), len(y), ntrial)
    assert_equal(len(mepochs), 100)
    assert_true(len(np.unique(mepochs.events[:, 2])), 100)
