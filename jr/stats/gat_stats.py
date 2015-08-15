import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy.stats import wilcoxon
# XXX something wrong with FDR: some scores are at 0?
from mne.stats import fdr_correction


def _loop(x, function):
    out = list()
    for ii in range(x.shape[1]):
        out.append(function(x[:, ii]))
    return out


def _my_wilcoxon(X):
    out = wilcoxon(X)
    return out[1]


def parallel_stats(X, function=_my_wilcoxon, correction='FDR', n_jobs=-1):
    from mne.parallel import parallel_func
    if correction not in [False, None, 'FDR']:
        raise ValueError('Unknown correction')
    # reshape to 2D
    X = np.array(X)
    dims = X.shape
    X.resize([dims[0], np.prod(dims[1:])])
    # prepare parallel
    n_cols = X.shape[1]
    parallel, pfunc, n_jobs = parallel_func(_loop, n_jobs)
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    p_values = parallel(pfunc(X[:, chunk], function) for chunk in chunks)
    p_values = np.reshape(np.hstack(p_values), dims[1:])
    X.resize(dims)
    # apply correction
    if correction == 'FDR':
        dims = p_values.shape
        _, p_values = fdr_correction(p_values)
        p_values = np.reshape(p_values, dims)
    return p_values


def _stat_fun(x, sigma=0, method='relative'):
    from mne.stats import ttest_1samp_no_p
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats_tfce(X, n_permutations=2**10,
               threshold=dict(start=.1, step=.1), n_jobs=2):
    X = np.array(X)
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
            X,
            out_type='mask',
            stat_fun=_stat_fun,
            n_permutations=n_permutations,
            threshold=threshold,
            n_jobs=n_jobs)
    p_values = p_values.reshape(X.shape[1:])
    return p_values
