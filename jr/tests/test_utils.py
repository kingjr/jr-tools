import numpy as np
from ..utils import tile_memory_free, pairwise


def test_tile_memory_free():
    from nose.tools import assert_equal
    y = np.arange(100)
    Y = tile_memory_free(y, 33)
    assert_equal(y.shape[0], Y.shape[0])
    np.testing.assert_array_equal(y, Y[:, 0], Y[:, -1])


def _dummy_pairwise_function_1(x, y):
    return x[0, :]


def _dummy_pairwise_function_2(x, y):
    return x[0, :], 0. * x[0, :]


def test_pairwise():
    from nose.tools import assert_equal, assert_raises
    n_obs = 20
    n_dims1 = 5
    n_dims2 = 10
    y = np.linspace(0, 1, n_obs)
    X = np.zeros((n_obs, n_dims1, n_dims2))
    for dim1 in range(n_dims1):
        for dim2 in range(n_dims2):
            X[:, dim1, dim2] = dim1 + 10*dim2

    # test size
    score = pairwise(X, y, _dummy_pairwise_function_1, n_jobs=2)
    assert_equal(score.shape, X.shape[1:])
    np.testing.assert_array_equal(score[:, 0], np.arange(n_dims1))
    np.testing.assert_array_equal(score[0, :], 10 * np.arange(n_dims2))

    # Test that X has not changed becaus of resize
    np.testing.assert_array_equal(X.shape, [n_obs, n_dims1, n_dims2])

    # test multiple out
    score1, score2 = pairwise(X, y, _dummy_pairwise_function_2, n_jobs=2)
    np.testing.assert_array_equal(score1[:, 0], np.arange(n_dims1))
    np.testing.assert_array_equal(score2[:, 0], 0 * np.arange(n_dims1))

    # Test array vs vector
    score1, score2 = pairwise(X, X, _dummy_pairwise_function_2, n_jobs=1)

    # test error check
    assert_raises(ValueError, pairwise, X, y[1:], _dummy_pairwise_function_1)
    assert_raises(ValueError, pairwise, y, X, _dummy_pairwise_function_1)
