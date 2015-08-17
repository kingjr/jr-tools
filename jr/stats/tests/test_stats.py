import numpy as np
from nose.tools import assert_almost_equal
from ..base import repeated_corr, repeated_spearman, corrcc


def test_corr_functions():
    from scipy.stats import spearmanr
    test_corr(np.corrcoef, repeated_corr, 1)
    test_corr(spearmanr, repeated_spearman, 0)


def test_corr(old_func, new_func, sel_item):
    from nose.tools import assert_equal, assert_raises
    n_obs = 20
    n_dims = 10
    y = np.linspace(0, 1, n_obs)
    X = np.tile(y, [n_dims, 1]).T + np.random.randn(n_obs, n_dims)
    rho_fast = new_func(X, y)
    # test dimensionality
    assert_equal(rho_fast.ndim, 1)
    assert_equal(rho_fast.shape[0], n_dims)
    # test data
    rho_slow = np.ones(n_dims)
    for dim in range(n_dims):
        rho_slow[dim] = np.array(old_func(X[:, dim], y)).item(sel_item)
    np.testing.assert_array_equal(rho_fast.shape, rho_slow.shape)
    np.testing.assert_array_almost_equal(rho_fast, rho_slow)
    # test errors
    new_func(np.squeeze(X[:, 0]), y)
    assert_raises(ValueError, new_func, y, X)
    assert_raises(ValueError, new_func, X, y[1:])
    # test dtype
    X = np.argsort(X, axis=0) * 2  # ensure no bug at normalization
    y = np.argsort(y, axis=0) * 2
    rho_fast = new_func(X, y, dtype=int)
    rho_slow = np.ones(n_dims)
    for dim in range(n_dims):
        rho_slow[dim] = np.array(old_func(X[:, dim], y)).item(sel_item)
    np.testing.assert_array_almost_equal(rho_fast, rho_slow)


def test_corrcc():
    import pycircstat
    np.random.seed(0)
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    assert_almost_equal(corrcc(x, y), pycircstat.corrcc(x, y))
