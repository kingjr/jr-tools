import numpy as np
from ..utils import tile_memory_free


def circ_hist(alpha, bins=None, n=100):
    if bins is None:
        bins = np.linspace(0, 2 * np.pi, n)
    h, bins = np.histogram(alpha % (2 * np.pi), bins)
    h = np.hstack((h, np.array(h[0])))
    bins = np.hstack((bins, np.array(bins[0])))
    return h, bins


def circ_tuning(alpha, bins=None, n=100):
    h, bins = circ_hist(alpha, bins, n=n)
    proba = 1. * h / np.sum(h)
    bins = bins - np.ptp(bins) / len(bins)
    bins = bins[1:]
    return proba, bins


def circ_double_tuning(radius, bins, wrapped=True):
    if wrapped:
        bins = np.hstack((bins[:-1] / 2, np.pi + bins[:-1] / 2, bins[0]))
        radius = np.hstack((radius[:-1], radius))
    else:
        bins = np.hstack((bins / 2, np.pi + bins / 2, bins[0]))
        radius = np.hstack((radius, radius, radius[0]))
    return radius, bins


def circ_diff(A, B):
    """WIP explodes memory"""
    raise NotImplementedError
    A = np.array(A) % (2 * np.pi)
    B = np.array(B) % (2 * np.pi)
    dims = A.shape
    if (B.ndim == 1) and (len(B) == 1):
        A = np.reshape([-1, 1])
        Bx = np.cos(B)
        By = np.sin(B)
        out = [np.angle(np.cos(a) - Bx + 1.j * (np.sin(a) - By)) for a in A]
    else:
        if len(A) != len(B):
            raise ValueError('A and B must share first axis')
        if (B.ndim == 1) and (A.ndim > 1):
            B = tile_memory_free(B, A.shape[0])
        out = [np.angle(np.cos(a) - np.cos(b) + 1.j * (np.sin(a) - np.sin(b)))
               for a, b in zip(A, B)]
    return out.reshape(dims)


def circ_mean(alpha, axis=None):
    alpha = np.arctan2(np.nanmean(np.sin(alpha), axis=axis),
                       np.nanmean(np.cos(alpha), axis=axis))
    return alpha % (2 * np.pi)


def circ_cummean(alpha, axis=None):
    alpha = np.arctan2(np.cummean(np.sin(alpha), axis=axis),
                       np.cummean(np.cos(alpha), axis=axis))
    return alpha % (2 * np.pi)


def circ_std(alpha, axis=None):
    return np.sqrt(np.nanmean(np.sin(alpha), axis=axis) ** 2 +
                   np.nanmean(np.cos(alpha), axis=axis) ** 2)
