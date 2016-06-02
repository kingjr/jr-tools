# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LinearSegmentedColormap
from ..utils import logcenter
from ..stats import median_abs_deviation

RdPuBu = col.LinearSegmentedColormap.from_list('RdPuBu', ['b', 'r'])


def share_clim(axes, clim=None):
    """Share clim across multiple axes
    Parameters
    ----------
    axes : plt.axes
    clim : np.array | list, shape(2,), optional
        Defaults is min and max across axes.clim.
    """
    # Find min max of clims
    if clim is None:
        clim = list()
        for ax in axes:
            for im in ax.get_images():
                clim += np.array(im.get_clim()).flatten().tolist()
        clim = [np.min(clim), np.max(clim)]
    # apply common clim
    for ax in axes:
        for im in ax.get_images():
            im.set_clim(clim)
    plt.draw()


def plot_widths(xs, ys, widths, ax=None, color='b', xlim=None, ylim=None,
                **kwargs):
    xs, ys, widths = np.array(xs), np.array(ys), np.array(widths)
    if not (len(xs) == len(ys) == len(widths)):
        raise ValueError('xs, ys, and widths must have identical lengths')
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)

    segmentx, segmenty = [xs[0]], [ys[0]]
    current_width = widths[0]
    for ii, (x, y, width) in enumerate(zip(xs, ys, widths)):
        segmentx.append(x)
        segmenty.append(y)
        if (width != current_width) or (ii == (len(xs) - 1)):
            ax.plot(segmentx, segmenty, linewidth=current_width, color=color,
                    **kwargs)
            segmentx, segmenty = [x], [y]
            current_width = width
    if xlim is None:
        xlim = [min(xs), max(xs)]
    if ylim is None:
        ylim = [min(ys), max(ys)]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax if fig is None else fig


def plot_sem(x, y, robust=False, **kwargs):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    robust : bool
        If False use mean + std,
        If True median + mad
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    x, y = np.array(x), np.array(y)
    if robust:
        m = np.median(y, axis=0)
        std = median_abs_deviation(y, axis=0)
    else:
        m = np.nanmean(y, axis=0)
        std = np.nanstd(y, axis=0)
    n = y.shape[0] - np.sum(np.isnan(y), axis=0)

    return plot_eb(x, m, std / np.sqrt(n), **kwargs)


def plot_eb(x, y, yerr, ax=None, alpha=0.3, color=None, line_args=dict(),
            err_args=dict()):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    yerr : list | np.array() | float
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    x, y = np.array(x), np.array(y)
    ax = ax if ax is not None else plt.gca()
    if 'edgecolor' not in err_args.keys():
        err_args['edgecolor'] = 'none'
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y,  color=color, **line_args)
    ax.fill_between(x, ymax, ymin, alpha=alpha, color=color, **err_args)

    return ax


def fill_betweenx_discontinuous(ax, ymin, ymax, x, freq=1, **kwargs):
    """Fill betwwen x even if x is discontinuous clusters
    Parameters
    ----------
    ax : axis
    x : list

    Returns
    -------
    ax : axis
    """
    x = np.array(x)
    min_gap = (1.1 / freq)
    while np.any(x):
        # If with single time point
        if len(x) > 1:
            xmax = np.where((x[1:] - x[:-1]) > min_gap)[0]
        else:
            xmax = [0]

        # If continuous
        if not np.any(xmax):
            xmax = [len(x) - 1]
        print x[0], x[xmax[0]]
        ax.fill_betweenx((ymin, ymax), x[0], x[xmax[0]], **kwargs)

        # remove from list
        x = x[(xmax[0] + 1):]
    return ax


def pcolormesh_45deg(C, ax=None, xticks=None, xticklabels=None, yticks=None,
                     yticklabels=None, aspect='equal', rotation=45,
                     *args, **kwargs):
    """Adapted from http://stackoverflow.com/questions/12848581/
    is-there-a-way-to-rotate-a-matplotlib-plot-by-45-degrees"""
    import itertools

    if ax is None:
        ax = plt.gca()
    n = C.shape[0]
    # create rotation/scaling matrix
    t = np.array([[1, .5], [-1, .5]])
    # create coordinate matrix and transform it
    product = itertools.product(range(n, -1, -1), range(0, n + 1, 1))
    A = np.dot(np.array([(ii[1], ii[0]) for ii in product]), t)
    # plot
    ax.pcolormesh((2 * A[:, 1].reshape(n + 1, n + 1) - n),
                  A[:, 0].reshape(n + 1, n + 1),
                  np.flipud(C), *args, **kwargs)

    xticks = np.linspace(0, n - 1, n, dtype=int) if xticks is None else xticks
    yticks = np.linspace(0, n - 1, n, dtype=int) if yticks is None else yticks

    if xticks is not None:
        xticklabels = xticks if xticklabels is None else xticklabels
        for tick, label, in zip(xticks, xticklabels):
            print tick, label
            ax.scatter(-n + tick + .5, tick + .5, marker='x', color='k')
            ax.text(-n + tick + .5, tick + .5, label,
                    horizontalalignment='right', rotation=-rotation)
    if yticks is not None:
        yticklabels = yticks if yticklabels is None else yticklabels
        for tick, label, in zip(yticks, yticklabels):
            ax.scatter(tick + .5, n - tick - .5, marker='x', color='k')
            ax.text(tick + .5, n - tick - .5, label,
                    horizontalalignment='left', rotation=rotation)

    if aspect:
        ax.set_aspect(aspect)
    ax.set_xlim(-n, n)
    ax.set_ylim(-n, n)
    ax.plot([-n, 0, n, 0., -n], [0, n, 0, -n, 0], color='k')
    ax.axis('off')
    return ax


def pretty_plot(ax=None):
    if ax is None:
        plt.gca()
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def pretty_colorbar(im=None, ax=None, ticks=None, ticklabels=None, nticks=3,
                    **kwargs):
    if ax is None:
        ax = plt.gca()
    if im is None:
        for obj in ax.get_children():
            if isinstance(obj, matplotlib.image.AxesImage):
                im = obj
                continue
        if im is None:
            raise RuntimeError('did not find the image')
    if ticks is None:
        clim = im.get_clim()
        # XXX bug https://github.com/matplotlib/matplotlib/issues/6352
        if None in clim:
            fig = ax.get_figure()
            fig.canvas.draw()
            clim = im.get_clim()
        ticks = np.linspace(clim[0], clim[1], nticks)
    cb = plt.colorbar(im, ax=ax, ticks=ticks, **kwargs)
    if ticklabels is None:
        ticklabels = ['%.2f' % ii for ii in ticks]
    cb.ax.set_yticklabels(ticklabels, color='dimgray')
    cb.ax.xaxis.label.set_color('dimgray')
    cb.ax.yaxis.label.set_color('dimgray')
    cb.ax.spines['left'].set_color('dimgray')
    cb.ax.spines['right'].set_color('dimgray')
    box = cb.ax.get_children()[2]
    box.set_edgecolor('dimgray')
    return cb


def get_datalim(ax):
    """WIP"""
    X, Y = [np.inf, -np.inf], [np.inf, -np.inf]
    for line in ax.lines:
        if not line.get_visible():
            continue
        x, y = line.get_data()
        X[0] = np.min(np.hstack((x, X[0])))
        X[1] = np.max(np.hstack((x, X[1])))
        Y[0] = np.min(np.hstack((y, Y[0])))
        Y[1] = np.max(np.hstack((y, Y[1])))
    for patch in ax.patches:
        if not patch.get_visible():
            continue
        x, y = patch.get_data()
        X[0] = np.min(np.hstack((x, X[0])))
        X[1] = np.max(np.hstack((x, X[1])))
        Y[0] = np.min(np.hstack((y, Y[0])))
        Y[1] = np.max(np.hstack((y, Y[1])))
    return X, Y


def share_lim(axes):
    """WIP"""
    X, Y = [np.inf, -np.inf], [np.inf, -np.inf]
    for ax in axes:
        x, y = get_datalim(ax)
        X[0] = np.min(np.hstack((x, X[0])))
        X[1] = np.max(np.hstack((x, X[1])))
        Y[0] = np.min(np.hstack((y, Y[0])))
        Y[1] = np.max(np.hstack((y, Y[1])))
    for ax in axes:
        ax.set_xlim(X[0], X[1])
        ax.set_ylim(Y[0], Y[1])
    return X, Y


def bar_sem(x, y=None, color='k', ax=None, bin_width=None, bottom=None,
            aplha=.5):
    if y is None:
        y = np.array(x)
        x = range(y.shape[1]) if y.ndim == 2 else range(len(y))
    if ax is None:
        ax = plt.gca()
    y, x = np.array(y), np.array(x)
    if (x.ndim > 1) or (x.shape[0] != y.shape[1]):
        raise ValueError('x and y must share first axis')
    if isinstance(color, str):
        color = [color] * len(x)
    elif isinstance(color, np.ndarray) and color.ndim == 1:
        color = [color] * len(x)
    means = np.nanmean(y, axis=0)
    sems = np.nanstd(y, axis=0) / np.sqrt(y.shape[0])
    if bin_width is None:
        bin_width = np.diff(x[:2])
    for mean, sem, bin_, this_color in zip(means, sems, x, color):
        options = dict(color=this_color, edgecolor='none', linewidth=0,
                       width=bin_width, bottom=bottom)
        ax.bar(bin_, mean + sem, alpha=aplha, **options)
        ax.bar(bin_, mean - sem, alpha=aplha, **options)
        ax.bar(bin_, mean, **options)
    pretty_plot(ax)
    return ax


class nonlinear_cmap(LinearSegmentedColormap):
    def __init__(self, cmap, center=.5, clim=[0, 1]):
        if isinstance(cmap, str):
            self.cmap = plt.get_cmap(cmap)
            self.clim = clim
            self.center = center
            for attr in self.cmap.__dict__.keys():
                setattr(self, attr, self.cmap.__dict__[attr])

    def __call__(self, value, alpha=1., **kwargs):
        center = (self.center - self.clim[0]) / np.diff(self.clim)
        value = (value - self.clim[0]) / np.diff(self.clim)
        value[value < 0] = 0.
        value[value > 1] = 1.
        ilogval = logcenter(center, x=value, inverse=True)
        return self.cmap(ilogval, alpha=alpha, **kwargs)


def pretty_axes(axes, xticks=None, xticklabels=None, yticks=None,
                yticklabels=None, xlabel=None, ylabel=None, xlabelpad=-10,
                ylabelpad=-10, xlim=None, ylim=None, aspect=None):
    ax0 = axes.reshape(-1)[0]
    fig = ax0.get_figure()
    fig.canvas.draw()
    xticks = ax0.get_xticks() if xticks is None else xticks
    xticklabels = [tick.get_text() for tick in ax0.get_xticklabels()] \
        if xticklabels is None else xticklabels
    xlabel = ax0.get_xlabel() if xlabel is None else xlabel
    xlim = ax0.get_xlim() if xlim is None else xlim
    yticks = ax0.get_yticks() if yticks is None else yticks
    yticklabels = [tick.get_text() for tick in ax0.get_yticklabels()] \
        if yticklabels is None else yticklabels
    ylabel = ax0.get_ylabel() if ylabel is None else ylabel
    ylim = ax0.get_ylim() if ylim is None else ylim
    aspect = ax0.get_aspect() if aspect is None else aspect
    if axes.ndim == 1:
        axes = np.reshape(axes, [1, len(axes)])
    n, m = axes.shape
    for ii in range(n):
        for jj in range(m):
            pretty_plot(axes[ii, jj])
            axes[ii, jj].set_xticks(xticks)
            axes[ii, jj].set_yticks(yticks)
            fig.canvas.draw()
            axes[ii, jj].set_xlim(xlim)
            axes[ii, jj].set_ylim(ylim)
            fig.canvas.draw()
            if ii != (n - 1):
                axes[ii, jj].set_xticklabels([''] * len(xticks))
                axes[ii, jj].set_xlabel('')
            else:
                axes[ii, jj].set_xticklabels(xticklabels)
                axes[ii, jj].set_xlabel(xlabel, labelpad=xlabelpad)
            if jj != 0:
                axes[ii, jj].set_yticklabels([''] * len(yticks))
                axes[ii, jj].set_ylabel('')
            else:
                axes[ii, jj].set_yticklabels(yticklabels)
                axes[ii, jj].set_ylabel(ylabel, labelpad=ylabelpad)
