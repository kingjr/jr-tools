# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import numpy as np
import matplotlib.pyplot as plt


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


def plot_sem(x, y, **kwargs):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
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


def pretty_plot(ax):
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
