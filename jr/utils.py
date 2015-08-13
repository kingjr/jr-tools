import sys
import numpy as np


def tile_memory_free(y, shape):
    """
    Tile vector along multiple dimension without allocating new memory.

    Parameters
    ----------
     y : np.array, shape (n,)
        data
    shape : np.array, shape (m),
    Returns
    -------
    Y : np.array, shape (n, *shape)
    """
    y = np.lib.stride_tricks.as_strided(y,
                                        (np.prod(shape), y.size),
                                        (0, y.itemsize)).T
    return y.reshape(np.hstack((len(y), shape)))
