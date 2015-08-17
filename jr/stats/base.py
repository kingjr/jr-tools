import numpy as np
from ..utils import tile_memory_free


def circular_linear_correlation(X, alpha):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #           Niccolo Pescetelli <niccolo.pescetelli@gmail.com>
    #
    # Licence : BSD-simplified
    """

    Parameters
    ----------
        X : numpy.array, shape (n_angles, n_dims)
            The linear data
        alpha : numpy.array, shape (n_angles,)
            The angular data (if n_dims == 1, repeated across all x dimensions)
    Returns
    -------
        R : numpy.array, shape (n_dims)
            R values
        R2 : numpy.array, shape (n_dims)
            R square values
        p_val : numpy.array, shape (n_dims)
            P values

    Adapted from:
        Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html
        Equantion 27.47
    """

    from scipy.stats import chi2
    import numpy as np

    # computes correlation for sin and cos separately
    rxs = repeated_corr(X, np.sin(alpha))
    rxc = repeated_corr(X, np.cos(alpha))
    rcs = repeated_corr(np.sin(alpha), np.cos(alpha))

    # tile alpha across multiple dimension without requiring memory
    if X.ndim > 1 and alpha.ndim == 1:
        rcs = tile_memory_free(rcs, X.shape[1:])

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)

    # JR adhoc way of having a sign....
    R = np.sign(rxs) * np.sign(rxc) * R
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(alpha)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def repeated_corr(X, y, dtype=float):
    """Computes pearson correlations between a vector and a matrix.

    Adapted from Jona-Sassenhagen's PR #L1772 on mne-python.

    Parameters
    ----------
        y : np.array, shape (n_samples)
            Data vector.
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    from sklearn.utils.extmath import fast_dot
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]
    y -= np.array(y.mean(0), dtype=dtype)
    X -= np.array(X.mean(0), dtype=dtype)
    y_sd = y.std(0, ddof=1)
    X_sd = X.std(0, ddof=1)[:, None if y.shape == X.shape else Ellipsis]
    return (fast_dot(y.T, X) / float(len(y) - 1)) / (y_sd * X_sd)


def repeated_spearman(X, y, dtype=None):
    """Computes spearman correlations between a vector and a matrix.

    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]

    # Rank
    X = np.argsort(X, axis=0)
    y = np.argsort(y, axis=0)
    # Double rank to ensure that normalization step of compute_corr
    # (X -= mean(X)) remains an integer.
    if (dtype is None and X.shape[0] < 2 ** 8) or\
       (dtype in [int, np.int16, np.int32, np.int64]):
        X *= 2
        y *= 2
        dtype = np.int16
    else:
        dtype = type(y[0])
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    return repeated_corr(X, y, dtype=type(y[0]))


def corrcc(alpha1, alpha2, axis=None):
    """ Circular correlation coefficient for two circular random variables.

    Adapted from pycircstat by Jean-Rémi King
    References: [Jammalamadaka2001]_
    """
    assert alpha1.shape == alpha2.shape, 'Input dimensions do not match.'

    # center data on circular mean
    def sin_center(alpha):
        m = np.arctan2(np.mean(np.sin(alpha), axis=axis),
                       np.mean(np.cos(alpha), axis=axis))
        return np.sin((alpha - m) % (2 * np.pi))

    sin_alpha1 = sin_center(alpha1)
    sin_alpha2 = sin_center(alpha2)

    # compute correlation coeffcient from p. 176
    num = np.sum(sin_alpha1 * sin_alpha2, axis=axis)
    den = np.sqrt(np.sum(sin_alpha1 ** 2, axis=axis) *
                  np.sum(sin_alpha2 ** 2, axis=axis))
    return num / den
