import numpy as np
from .utils import tile_memory_free


def test_tile_memory_free():
    from nose.tools import assert_equal
    y = np.arange(100)
    Y = tile_memory_free(y, 33)
    assert_equal(y.shape[0], Y.shape[0])
    np.testing.assert_array_equal(y, Y[:, 0], Y[:, -1])
