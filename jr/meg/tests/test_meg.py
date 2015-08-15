import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_true, assert_raises
from ..base import mat2mne


def test_mat2mne():
    ntrial, nchan, ntime = 100, 32, 10
    data = np.random.randn(ntrial, nchan, ntime)
    # test base
    epochs = mat2mne(data)
    # test dimensionalities
    assert_array_equal(epochs._data.shape, [ntrial, nchan, ntime])
    assert_equal(epochs._data.shape[0], len(epochs.events), ntrial)
    # test chan names
    assert_true(len(np.unique(epochs.ch_names)) == nchan)
    epochs = mat2mne(data, chan_names='eeg')
    assert_true(len(np.unique(epochs.ch_names)) == nchan)
    # test sfreq
    epochs = mat2mne(data, sfreq=1000)
    assert_equal(epochs.info['sfreq'], 1000)
    # test events
    events = np.round(np.random.rand(ntrial))
    epochs = mat2mne(data, events=events)
    assert_array_equal(epochs.events[:, 2], events)
    events = np.round(np.random.rand(ntrial, 3))
    epochs = mat2mne(data, events=events)
    assert_array_equal(epochs.events, events)
    assert_raises(data, events=1)
    assert_raises(data, events=np.ones(ntrial + 1))
    assert_raises(data, events=np.ones((10, 2)))
    assert_raises(data, events=np.ones((10, 2, 3, 4)))
