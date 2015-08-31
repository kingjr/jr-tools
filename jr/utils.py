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
