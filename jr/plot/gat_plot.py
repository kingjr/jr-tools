import matplotlib.pyplot as plt
import numpy as np
from .base import pretty_plot, plot_sem, plot_widths


def pretty_gat(scores, times, chance, ax=None, sig=None, cmap='RdBu_r',
               clim=None, colorbar=True, xlabel='Testing Times (s.)',
               ylabel='Train times (s.)'):
    scores = np.array(scores)

    # setup color range
    if clim is None:
        spread = 2 * np.round(np.percentile(
            np.abs(scores - chance), 99) * 1e2) / 1e2
        m = chance
        vmin, vmax = m + spread * np.array([-.6, .6])
    elif len(clim) == 1:
        vmin, vmax = clim - chance, clim
    else:
        vmin, vmax = clim

    # setup time
    test_times = np.cumsum(
        np.ones(scores.shape[1])) * np.ptp(times) / len(times)
    extent = [min(test_times), max(test_times), min(times), max(times)]

    # setup plot
    if ax is None:
        ax = plt.gca()

    # plot score
    im = ax.imshow(scores, extent=extent, cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, aspect='equal')

    # plot sig
    if sig is not None:
        sig = np.array(sig)
        xx, yy = np.meshgrid(test_times, times, copy=False, indexing='xy')
        ax.contour(xx, yy, sig, colors='black', levels=[0],
                   linestyles='dotted')
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    if colorbar:
        cb = plt.colorbar(im, ax=ax, ticks=[vmin, chance, vmax])
        cb.ax.set_yticklabels(['%.2f' % vmin, 'Chance', '%.2f' % vmax],
                              color='dimgray')
        cb.ax.xaxis.label.set_color('dimgray')
        cb.ax.yaxis.label.set_color('dimgray')
        cb.ax.spines['left'].set_color('dimgray')
        cb.ax.spines['right'].set_color('dimgray')
        box = cb.ax.get_children()[2]
        box.set_edgecolor('dimgray')

    # setup ticks
    xticks, xticklabels = _set_ticks(test_times)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    yticks, yticklabels = _set_ticks(times)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    yticks = np.arange(min(times), max(times), .100)
    yticks = np.round(xticks * 10.) / 10.
    if len(xlabel):
        ax.set_xlabel(xlabel)
    if len(ylabel):
        ax.set_ylabel(ylabel)
    pretty_plot(ax)
    return ax


def pretty_decod(scores, times, chance, ax=None, sig=None, width=3.,
                 color='k', fill=False, xlabel='Times (s.)'):
    scores = np.array(scores)

    # setup plot
    if ax is None:
        ax = plt.gca()

    # Plot SEM
    if scores.ndim == 2:
        scores_m = np.mean(scores, axis=0)
        sem = scores.std(0) / np.sqrt(len(scores))
        plot_sem(times, scores, color=color, ax=ax)
    else:
        scores_m = scores
        sem = np.zeros_like(scores_m)

    # Plot significance
    if sig is not None:
        sig = np.array(sig)
        widths = width * sig
        plot_widths(times, scores_m, widths, ax=ax, color=color)
        if fill:
            scores_sig = (chance + (scores_m - chance) * sig)
            ax.fill_between(times, chance, scores_sig, color=color,
                            alpha=.75, linewidth=0)

    # Pretty
    ymin, ymax = min(scores_m - sem), max(scores_m + sem)
    ax.axhline(chance, linestyle='dotted', color='k', zorder=-3)
    ax.axvline(0, color='k', zorder=-3)
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([ymin, chance, ymax])
    ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax])
    xticks, xticklabels = _set_ticks(times)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    if len(xlabel):
        ax.set_xlabel(xlabel)
    pretty_plot(ax)
    return ax


def _set_ticks(times):
    ticks = np.arange(min(times), max(times), .100)
    if np.round(max(times) * 10.) / 10. == max(times):
        ticks = np.append(ticks, max(times))
    ticks = np.round(ticks * 10.) / 10.
    ticklabels = ([int(ticks[0] * 1e3)] +
                  ['' for ii in ticks[1:-1]] +
                  [int(ticks[-1] * 1e3)])
    return ticks, ticklabels
