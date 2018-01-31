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
    XXX Will be deprecated
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
    import warnings
    warnings.warn('Will be deprecated. Use np.newaxis instead')
    for dim in range(len(shape)):
        y = y[..., np.newaxis]
    return y


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
        self._parse_script(section)
        if not isinstance(fig, list):
            fig = [fig]
            title = [title]
        for this_fig, this_title in zip(fig, title):
            fname = op.join(self.report.data_path,
                            section + '_' + this_title + '.png')
            this_fig.savefig(fname, transparent=True, dpi=200)

            fname = op.join(self.report.data_path,
                            section + '_' + this_title + '.ps')
            this_fig.savefig(fname)
        return self.report.add_figs_to_section(fig, title, section)

    def add_images_to_section(self, fig, title, section):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        self._parse_script(section)
        return self.report.add_images_to_section(fig, title, section)

    def add_htmls_to_section(self, html, title, section):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        self._parse_script(section)
        return self.report.add_htmls_to_section(html, title, section)

    def save(self, open_browser=None):
        if not hasattr(self, 'report'):
            self._setup_provenance()
        if open_browser is None:
            open_browser = not self.use_agg
        # add script
        html = self._py_to_html(self.pyscript)
        self.report.add_htmls_to_section(html, 'script', 'script')
        self.report.save(open_browser=open_browser)
        if self.upload_on_save is True and self.client is not None:
            self.client.upload(self.report.data_path)

    def _parse_script(self, section):
        return
        # WIP XXX TODO
        import re
        from inspect import getouterframes, currentframe
        # get caller line number
        curr_call_line = getouterframes(currentframe())[1][2]
        # find 'report.' in script
        calls = np.array([m.start()
                          for m in re.finditer('report\.', self.pyscript)])
        # identify line number
        lines = np.array([m.start()
                          for m in re.finditer('\n', self.pyscript)])
        # get line number for each 'report.' call
        calls_line = [sum(call > lines) for call in calls]
        # identify line of previous 'report.call'
        prev_call_line = np.where(calls_line < curr_call_line)[0]
        prev_call_line = 0 if not len(prev_call_line) else prev_call_line
        # capture cell
        line_start = calls_line[prev_call_line]
        line_stop = calls_line[curr_call_line]
        cell = self.pyscript(lines[line_start:line_stop])
        # get html
        html = self._py_to_html(cell)
        self.report.add_htmls_to_section(html, 'script', section)

    def _py_to_html(self, txt):
        html = highlight(txt, PythonLexer(),
                         HtmlFormatter(linenos=True, cssclass="source"))
        return html


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
    Y = X * v[:, np.newaxis]
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


def align_on_diag(matrix, center=False):
    matrix = np.array(matrix)
    n, m = matrix.shape[:2]
    if n != m:
        raise ValueError('matrix must be square')
    for ii in range(n):
        this_slice = np.array(range(ii, n) + range(0, ii))
        matrix[ii, :, ...] = matrix[ii, (n / 2 + this_slice) % n, ...]
    if not center:
        matrix = np.concatenate((matrix[:, (n // 2):, ...],
                                 matrix[:, :(n // 2), ...]), axis=1)
    return matrix


def table2html(array, head_column=None, head_line=None,
               border=1):
    html = '<TABLE border="%i">' % border
    if head_line is not None:
        array = np.hstack((np.array([str(h) for h in head_line])[:, None],
                          array))
    if head_column is not None:
        head_column = [str(h) for h in head_column]
        if head_line is not None:
            head_column = np.hstack(([''], head_column))
        array = np.vstack((np.array(head_column)[None, :], array))
    for ii, line in enumerate(array):
        html += '<TR>'
        for jj, column in enumerate(line):
            html += '<TD>%s</TD>' % array[ii, jj]
        html += '</TR>'
    html += '</TABLE>'
    return html


def regular_split(X, n_split, axis=0):
    """ Split array similarly to np.array_split but crop axis to ensure
    regular splits."""
    X = np.asarray(X)

    # transpose to apply on first axis
    if axis != 0:
        dims = [axis] + range(0, axis) + range(axis + 1, X.ndim)
        X = np.transpose(X, dims)
    n = len(X)
    idx_max = n - n % n_split
    X_split = np.array_split(X[:idx_max, ...], n_split, axis=0)

    if axis != 0:
        inv_dims = range(1, axis + 1) + [0] + range(axis + 1, X.ndim)
        X_split = [x.transpose(inv_dims) for x in X_split]
    return np.array(X_split)


def match_list(A, B, on_replace='delete'):
    """Match two lists of different sizes and return corresponding indice

    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list

    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """
    from Levenshtein import editops

    A = np.nan_to_num(np.squeeze(A))
    B = np.nan_to_num(np.squeeze(B))
    assert A.ndim == B.ndim == 1

    unique = np.unique(np.r_[A, B])
    label_encoder = dict((k, v) for v, k in enumerate(unique))

    def int_to_unicode(array):
        return ''.join([str(chr(label_encoder[ii])) for ii in array])

    changes = editops(int_to_unicode(A), int_to_unicode(B))
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type, val_a, val_b in changes:
        if type == 'insert':
            B_sel[val_b] = np.nan
        elif type == 'delete':
            A_sel[val_a] = np.nan
        elif on_replace == 'delete':
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == 'keep':
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


if __file__ == '__main__':

    def test(A, B, on_replace, expected_length, expected_diff):
        A = np.asarray(A)
        B = np.asarray(B)
        a, b = match_list(A, B, on_replace)
        assert expected_length == len(a)
        assert expected_diff == sum(A[a] != B[b])

    for on_replace in ('delete', 'keep', None):
        test([10, 11, 12], [10, 11, 12], on_replace, 3, 0)
        test([10, 11], [10, 11, 12, 13], on_replace, 2, 0)
        test([10, 11, 12, 13], [10, 12], on_replace, 2, 0)
        test(range(0, 20), range(5, 25), on_replace, 15, 0)

    test([10, 99, 12], [10, 11, 12], 'delete', 2, 0)
    test([10, 11, 12], [10, 99, 12], 'keep', 3, 1)
