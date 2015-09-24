import os.path as op
import numpy as np
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


def pi_labels(X):
    from numpy import pi
    if len(np.shape(X)) == 0:
        X = [X]
    d = {0: r'0', pi: r'$\pi$', 2*pi: r'$2\pi$', pi/2: r'$\pi/2$',
         3*pi/2: r'$3\pi/2$', pi/4: r'$\pi/4$', 3*pi/4: r'$3\pi/4$',
         pi/3: r'$\pi/3$'}
    labels = list()
    for x in X:
        if isinstance(x, str):
            labels.append(x)
        elif x in d.keys():
            labels.append(d[x])
        elif -x in d.keys():
            labels.append(r'-' + d[-x])
        else:
            labels.append(r'%.2f' % x)
    return labels


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
    y = np.lib.stride_tricks.as_strided(np.array(y),
                                        (np.prod(shape), y.size),
                                        (0, y.itemsize)).T
    return y.reshape(np.hstack((len(y), shape)))


class OnlineReport():
    def __init__(self, script='config.py', client=None, upload_on_save=True,
                 results_dir='results/', use_agg=None):
        """WIP"""
        import inspect
        # setup path according to highest script
        for item in inspect.stack():
            if (
                item and (item[1][0] != '<') and
                ('python2.7' not in item[1]) and
                ('utils.py' not in item[1]) and
                ('config.py' not in item[1])
            ):
                script = item[1]
                break
        self.results_dir = results_dir
        self.script = script
        self.use_agg = use_agg
        self.upload_on_save = upload_on_save
        self.client = client

    def _setup_provenance(self):
        import os
        from meeg_preprocessing.utils import setup_provenance
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        # Read script
        with open(self.script, 'rb') as f:
            self.pyscript = f.read()
        # Setup display environment
        if self.use_agg is None:
            self.use_agg = os.getenv('use_agg')
            if self.use_agg:
                self.use_agg = True
        # Create report
        self.report, self.run_id, self.results_dir, self.logger = \
            setup_provenance(self.script, self.results_dir,
                             use_agg=self.use_agg)

    def add_figs_to_section(self, fig, title, section):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        if not isinstance(fig, list):
            fig = [fig]
            title = [title]
        for this_fig, this_title in zip(fig, title):
            fname = op.join(self.report.data_path,
                            section + '_' + this_title + '.png')
            this_fig.savefig(fname, transparent=True, dpi=200)
        return self.report.add_figs_to_section(fig, title, section)

    def add_images_to_section(self, fig, title, section):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        return self.report.add_images_to_section(fig, title, section)

    def save(self, open_browser=None):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        if open_browser is None:
            open_browser = not self.use_agg
        # add script
        html = highlight(self.pyscript, PythonLexer(),
                         HtmlFormatter(linenos=True, cssclass="source"))
        self.report.add_htmls_to_section(html, 'script', 'script')
        self.report.save(open_browser=open_browser)
        if self.upload_on_save is True and self.client is not None:
            self.client.upload(self.report.data_path, self.report.data_path)


def nandigitize(x, bins, right=None):
    x = np.array(x)
    dims = x.shape
    x = np.reshape(x, [1, -1])
    sel = ~ np.isnan(x)
    x_ = x[sel]
    x[sel] = np.digitize(x_, bins, right=right)
    x[~sel] = np.nan
    return x.reshape(dims)


def count(x):
    return {ii: sum(x == ii) for ii in np.unique(x)}


def product_matrix_vector(X, v, axis=0):
    """
    Computes product between a matrix and a vector
    Input:
    ------
    X : np.array, shape[axis] = n
        The matrix
    v : np.array, shape (n)
        The vector
    axis : int
        The axis.
    Returns
    -------
    Y : np.array, shape == X.shape
    """
    # ensure numpy array
    X = np.array(X)
    v = np.squeeze(v)
    # ensure axis = 0
    if axis != 0:
        X = np.transpose(X, [[axis] + range(axis) + range(axis + 1, X.ndim)])
    if X.shape[0] != v.shape[0]:
        raise ValueError('X and v shapes must be identical on the chosen axis')
    # from nD to 2D
    dims = X.shape
    if X.ndim != 2:
        X = np.reshape(X, [X.shape[0], -1])
    # product
    # rows, columns = X.shape
    # Y = np.zeros((rows, columns))
    # for jj, m in enumerate(X.T):
    #     Y[:, jj] = m * v
    V = tile_memory_free(v, X.shape[1])
    Y = X * V
    # from 2D to nD
    if X.ndim != 2:
        Y = np.reshape(Y, dims)
    # transpose axes back to original
    if axis != 0:
        Y = Y.transpose([range(1, axis) + [axis] + range(axis + 1, X.ndim)])
    return Y


def logcenter(center, x=None, inverse=False):
    # Authors: Jean-Remi King, <jeanremi.king@gmail.com>
    #          Clement Levrard <clement.levrard@gmail.com>
    #
    # License: BSD (3-clause)
    """
    Creates a logarithmic scale centered around center, and bounded between
    [0., 1.] such that:
        f(0, center) = 0
        f(1, center) = 1
        f(center, center) = .5

    Parameters
    ----------
        x : float | np.array | None
            If float or np.array, 0. < x < 1.
            If None, set to np.linspace(0., 1., 256).
            Defaults to None.
        center : float
            0. < center < 1.
    Returns
    -------
        y : float | np.array
    """

    from numpy import exp, log
    if x is None:
        x = np.linspace(0., 1., 256)
    if center >= 1. or center <= 0.:
        raise ValueError('center must be between 0 and 1')
    if center == .5:
        y = x
    else:
        n = 1. / center
        if inverse is False:
            y = (exp(2 * log(n - 1) * x) - 1) / (n * (n - 2))
        else:
            y = log(x * (n * (n - 2)) + 1) / (2 * log(n - 1))
    if center > .5:
        y = 1. - y
    return y


def pairwise(X, y, func, n_jobs=-1):
    """Applies pairwise operations on two matrices using multicore:
    function(X[:, jj, kk, ...], y[:, jj, kk, ...])

    Parameters
    ----------
        X : np.ndarray, shape(n, ...)
        y : np.array, shape(n, ...) | shape(n,)
            If shape == X.shape:
                parallel(X[:, chunk], y[:, chunk ] for chunk in n_chunks)
            If shape == X.shape[0]:
                parallel(X[:, chunk], y for chunk in n_chunks)
        func : function
        n_jobs : int, optional
            Number of parallel cpu.
    Returns
    -------
        out : np.array, shape(func(X, y))
    """
    import numpy as np
    from mne.parallel import parallel_func
    dims = X.shape
    if y.shape[0] != dims[0]:
        raise ValueError('X and y must have identical shapes')

    X.resize([dims[0], np.prod(dims[1:])])
    if y.ndim > 1:
        Y = np.reshape(y, [dims[0], np.prod(dims[1:])])

    parallel, pfunc, n_jobs = parallel_func(func, n_jobs)

    n_cols = X.shape[1]
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    if y.ndim == 1:
        out = parallel(pfunc(X[:, chunk], y) for chunk in chunks)
    else:
        out = parallel(pfunc(X[:, chunk], Y[:, chunk]) for chunk in chunks)

    # size back in case higher dependencies
    X.resize(dims)

    # unpack
    if isinstance(out[0], tuple):
        return [np.reshape(out_, dims[1:]) for out_ in zip(*out)]
    else:
        return np.reshape(np.hstack(out), dims[1:])


def resample2D(x):
    """WIP"""
    factor = 5.
    x = x[:, None].T if x.ndim == 1 else x
    x = x[:, :, None].T if x.ndim == 2 else x
    this_range = range(int(np.floor(x.shape[1] / factor) * factor))
    x = x[:, this_range, :]
    x = x[:, :, this_range]
    x_list = list()
    for t in range(x.shape[1]):
        x_ = np.reshape(x[:, t, :], [x.shape[0], x.shape[2] / factor, factor])
        x_list.append(np.mean(x_, axis=2))
    x = np.transpose(x_list, [1, 2, 0])
    x = np.reshape(x_list, [x.shape[0], x.shape[1], x.shape[2] / factor,
                            factor])
    x = np.mean(x, axis=3)
    x = np.array([np.diag(ii) for ii in x])
    return x


def align_on_diag(matrix):
    matrix = np.array(matrix)
    n, m = matrix.shape[:2]
    if n != m:
        raise ValueError('matrix must be square')
    for ii in range(n):
        this_slice = np.array(range(ii, n) + range(0, ii))
        matrix[ii, :, ...] = matrix[ii, (n / 2 + this_slice) % n, ...]
    return matrix
