import numpy as np
from ..utils import tile_memory_free, product_matrix_vector


def corr_linear_circular(X, alpha):
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


def corr_circular_linear(alpha, X):
    # Authors:  Jean-Remi King <jeanremi.king@gmail.com>
    #
    # Licence : BSD-simplified
    """

    Parameters
    ----------
        alpha : numpy.array, shape (n_angles,)
            The angular data (if n_dims == 1, repeated across all x dimensions)
        X : numpy.array, shape (n_angles, n_dims)
            The linear data
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
    from jr.utils import pairwise
    import numpy as np

    # computes correlation for sin and cos separately
    # WIP Applies repeated correlation if X is vector
    # TODO: deals with non repeated correlations (X * ALPHA)
    if alpha.ndim > 1:
        rxs = repeated_corr(np.sin(alpha), X)
        rxc = repeated_corr(np.cos(alpha), X)
        rcs = np.zeros_like(alpha[0, :])
        rcs = pairwise(np.sin(alpha), np.cos(alpha), func=_loop_corr,
                       n_jobs=-1)
    else:
        # WIP Applies repeated correlation if alpha is vector
        rxs = repeated_corr(X, np.sin(alpha))
        rxc = repeated_corr(X, np.cos(alpha))
        rcs = repeated_corr(np.sin(alpha), np.cos(alpha))

    # Adapted from equation 27.47
    R = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2)

    # JR adhoc way of having a sign....
    R = np.sign(rxs) * np.sign(rxc) * R
    R2 = np.sqrt(R ** 2)

    # Get degrees of freedom
    n = len(X)
    pval = 1 - chi2.cdf(n * R2, 2)

    return R, R2, pval


def _loop_corr(X, Y):
    R = np.zeros(X.shape[1])
    for ii, (x, y) in enumerate(zip(X.T, Y.T)):
        R[ii] = repeated_corr(x, y)
    return R


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
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')
    if X.ndim == 1:
        X = X[:, None]
    ym = np.array(y.mean(0), dtype=dtype)
    Xm = np.array(X.mean(0), dtype=dtype)
    y -= ym
    X -= Xm
    y_sd = y.std(0, ddof=1)
    X_sd = X.std(0, ddof=1)[:, None if y.shape == X.shape else Ellipsis]
    R = (fast_dot(y.T, X) / float(len(y) - 1)) / (y_sd * X_sd)
    R = np.reshape(R, shape[1:])
    # cleanup variable changed in place
    y += ym
    X += Xm
    return R


def repeated_spearman(X, y, dtype=None):
    """Computes spearman correlations between a vector and a matrix.

    Parameters
    ----------
        X : np.array, shape (n_samples, n_measures ...)
            Data matrix onto which the vector is correlated.
        y : np.array, shape (n_samples)
            Data vector.
        dtype : type, optional
            Data type used to compute correlation values to optimize memory.

    Returns
    -------
        rho : np.array, shape (n_measures)
    """
    from scipy.stats import rankdata
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.ndim == 1:
        X = X[:, None]
    shape = X.shape
    X = np.reshape(X, [shape[0], -1])
    if X.ndim not in [1, 2] or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError('y must be a vector, and X a matrix with an equal'
                         'number of rows.')

    # Rank
    X = np.apply_along_axis(rankdata, 0, X)
    y = np.apply_along_axis(rankdata, 0, y)
    # Double rank to ensure that normalization step of compute_corr
    # (X -= mean(X)) remains an integer.
    X *= 2
    y *= 2
    X = np.array(X, dtype=dtype)
    y = np.array(y, dtype=dtype)
    R = repeated_corr(X, y, dtype=type(y[0]))
    R = np.reshape(R, shape[1:])
    return R


def corr_circular(ALPHA1, alpha2, axis=0):
    """ Circular correlation coefficient for two circular random variables.
    Input:
    ------
    ALPHA1 : np.array, shape[axis] = n
        The matrix
    alpha2 : np.array, shape (n), or shape == ALPHA1.shape
        Vector or matrix
    axis : int
        The axis used to estimate correlation
    Returns
    -------
    Y : np.array, shape == X.shape

    Adapted from pycircstat by Jean-Remi King :
    1. Less memory consuming than original
    2. supports ALPHA1 as matrix and alpha2 as vector
    https://github.com/circstat/pycircstat
    References: [Jammalamadaka2001]_
    """

    # center data on circular mean
    def sin_center(alpha):
        m = np.arctan2(np.mean(np.sin(alpha), axis=axis),
                       np.mean(np.cos(alpha), axis=axis))
        return np.sin((alpha - m) % (2 * np.pi))

    sin_alpha1 = sin_center(ALPHA1)
    sin_alpha2 = sin_center(alpha2)

    # compute correlation coeffcient from p. 176
    if sin_alpha1.ndim == sin_alpha2.ndim:
        num = np.sum(sin_alpha1 * sin_alpha2, axis=axis)
        den = np.sqrt(np.sum(sin_alpha1 ** 2, axis=axis) *
                      np.sum(sin_alpha2 ** 2, axis=axis))
    else:
        num = np.sum(product_matrix_vector(sin_alpha1, sin_alpha2, axis=axis))
        den = np.sqrt(np.sum(sin_alpha1 ** 2, axis=axis) *
                      np.sum(sin_alpha2 ** 2))
    return num / den


def robust_mean(X, axis=None, percentile=[5, 95]):
    X = np.array(X)
    axis_ = axis
    # force axis to be 0 for facilitation
    if axis is not None and axis != 0:
        X = np.transpose(X, [axis] + range(0, axis) + range(axis+1, X.ndim))
        axis_ = 0
    mM = np.percentile(X, percentile, axis=axis_)
    indices_min = np.where(X < np.tile(mM[0], [X.shape[0], 1, 1]))
    indices_max = np.where(X > np.tile(mM[1], [X.shape[0], 1, 1]))
    X[indices_min] = np.nan
    X[indices_max] = np.nan
    m = np.nanmean(X, axis=axis_)
    return m


def fast_mannwhitneyu(X, Y, use_continuity=True, n_jobs=-1):
    from mne.parallel import parallel_func
    X = np.array(X)
    Y = np.array(Y)
    nx, ny = len(X), len(Y)
    dims = X.shape
    X = np.reshape(X, [nx, -1])
    Y = np.reshape(Y, [ny, -1])
    parallel, p_time_gen, n_jobs = parallel_func(_loop_mannwhitneyu, n_jobs)
    n_chunks = np.min([n_jobs, X.shape[1]])
    chunks = np.array_split(range(X.shape[1]), n_chunks)
    out = parallel(p_time_gen(X[..., chunk],
                              Y[..., chunk], use_continuity=use_continuity)
                   for chunk in chunks)
    # Unpack estimators into time slices X folds list of lists.
    U, p_value = map(list, zip(*out))
    U = np.concatenate(U, axis=1).reshape(dims[1:])
    p_value = np.concatenate(p_value, axis=1).reshape(dims[1:])
    AUC = U / (nx * ny)
    # correct directionality of U stats imposed by mannwhitneyu
    if nx > ny:
        AUC = 1 - AUC
    return U, p_value, AUC


def _loop_mannwhitneyu(X, Y, use_continuity=True):
    n_col = X.shape[1]
    U, P = np.zeros(n_col), np.zeros(n_col)
    for ii in range(n_col):
        U[ii], P[ii] = mannwhitneyu(X[:, ii], Y[:, ii], use_continuity)
    return U, P


def dPrime(hits, misses, fas, crs):
    from scipy.stats import norm
    from math import exp, sqrt
    Z = norm.ppf
    hits, misses, fas, crs = float(hits), float(misses), float(fas), float(crs)
    # From Jonas Kristoffer Lindelov : lindeloev.net/?p=29
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5 / (hits + misses)
    halfFa = 0.5 / (fas + crs)

    # Calculate hitrate and avoid d' infinity
    hitRate = hits / (hits + misses)
    if hitRate == 1:
        hitRate = 1 - halfHit
    if hitRate == 0:
        hitRate = halfHit

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas/(fas+crs)
    if faRate == 1:
        faRate = 1 - halfFa
    if faRate == 0:
        faRate = halfFa

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp(Z(faRate)**2 - Z(hitRate)**2)/2
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out


def mannwhitneyu(x, y, use_continuity=True):
    """Adapated from scipy.stats.mannwhitneyu but includes direction of U"""
    from scipy.stats._rank import rankdata, tiecorrect
    from scipy.stats import distributions
    from numpy import asarray
    x = asarray(x)
    y = asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1*n2 - u1  # remainder is U for y
    T = tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in amannwhitneyu')
    sd = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)

    if use_continuity:
        # normal approximation for prob calc with continuity correction
        z = abs((u1 - 0.5 - n1*n2/2.0) / sd)
    else:
        z = abs((u1 - n1*n2/2.0) / sd)  # normal approximation for prob calc

    return u2, distributions.norm.sf(z)


def nested_analysis(X, df, condition, function=None, query=None,
                    single_trial=False, y=None, n_jobs=-1):
    """ Apply a nested set of analyses.
    Parameters
    ----------
    X : np.array, shape(n_samples, ...)
        Data array.
    df : pandas.DataFrame
        Condition DataFrame
    condition : str | list
        If string, get the samples for each unique value of df[condition]
        If list, nested call nested_analysis.
    query : str | None, optional
        To select a subset of trial using pandas.DataFrame.query()
    function : function
        Computes across list of evoked. Must be of the form:
        function(X[:], y[:])
    y : np.array, shape(n_conditions)
    n_jobs : int
        Number of core to compute the function. Defaults to -1.

    Returns
    -------
    scores : np.array, shape(...)
        The results of the function
    sub : dict()
        Contains results of sub levels.
    """
    import numpy as np
    from jr.utils import pairwise
    if isinstance(condition, str):
        # Subselect data using pandas.DataFrame queries
        sel = range(len(X)) if query is None else df.query(query).index
        X = X.take(sel, axis=0)
        y = np.array(df[condition][sel])
        # Find unique conditions
        values = list()
        for ii in np.unique(y):
            if (ii is not None) and (ii not in [np.nan]):
                try:
                    if np.isnan(ii):
                        continue
                    else:
                        values.append(ii)
                except TypeError:
                    values.append(ii)
        # Subsubselect for each unique condition
        y_sel = [np.where(y == value)[0] for value in values]
        # Mean condition:
        X_mean = np.zeros(np.hstack((len(y_sel), X.shape[1:])))
        y_mean = np.zeros(len(y_sel))
        for ii, sel_ in enumerate(y_sel):
            X_mean[ii, ...] = np.mean(X[sel_, ...], axis=0)
            if isinstance(y[sel_[0]], str):
                y_mean[ii] = ii
            else:
                y_mean[ii] = y[sel_[0]]
        if single_trial:
            X = X.take(np.hstack(y_sel), axis=0)  # ERROR COME FROM HERE
            y = y.take(np.hstack(y_sel), axis=0)
        else:
            X = X_mean
            y = y_mean
        # Store values to keep track
        sub_list = dict(X=X_mean, y=y_mean, sel=sel, query=query,
                        condition=condition, values=values,
                        single_trial=single_trial)
    elif isinstance(condition, list):
        # If condition is a list, we must recall the function to gather
        # the results of the lower levels
        sub_list = list()
        X_list = list()  # FIXME use numpy array
        for subcondition in condition:
            scores, sub = nested_analysis(
                X, df, subcondition['condition'], n_jobs=n_jobs,
                function=subcondition.get('function', None),
                query=subcondition.get('query', None))
            X_list.append(scores)
            sub_list.append(sub)
        X = np.array(X_list)
        if y is None:
            y = np.arange(len(condition))
        if len(y) != len(X):
            raise ValueError('X and y must be of identical shape: ' +
                             '%s <> %s') % (len(X), len(y))
        sub_list = dict(X=X, y=y, sub=sub_list, condition=condition)

    # Default function
    function = _default_analysis if function is None else function

    scores = pairwise(X, y, function, n_jobs=n_jobs)
    return scores, sub_list


def _default_analysis(X, y):
    # from sklearn.metrics import roc_auc_score
    from jr.stats import fast_mannwhitneyu
    # Binary contrast
    unique_y = np.unique(y)
    # if two condition, can only return contrast
    if len(y) == 2:
        y = np.where(y == unique_y[0], 1, -1)
        # Tile Y to across X dimension without allocating memory
        Y = tile_memory_free(y, X.shape[1:])
        return np.mean(X * Y, axis=0)
    elif len(unique_y) == 2:
        # if two conditions but multiple trials, can return AUC
        # auc = np.zeros_like(X[0])
        _, _, auc = fast_mannwhitneyu(X[y == unique_y[0], ...],
                                      X[y == unique_y[1], ...], n_jobs=1)
        # for ii, x in enumerate(X.T):
        #     auc[ii] = roc_auc_score(y, np.copy(x))
        return auc
    # Linear regression:
    elif len(unique_y) > 2:
        return repeated_spearman(X, y)
    else:
        raise RuntimeError('Please specify a function for this kind of data')


def median_abs_deviation(x, axis=None):
    """median absolute deviation"""
    x = np.array(x)
    shape = np.shape(x)
    center = np.median(x, axis=axis, keepdims=True)
    if axis is not None:
        tile = np.ones(len(shape))
        tile[axis] = shape[axis]
        center = np.tile(center, tile)
    return np.median(np.abs(x - center), axis=axis)
