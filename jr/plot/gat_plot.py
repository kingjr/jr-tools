import matplotlib.pyplot as plt
import numpy as np
from .base import pretty_plot, plot_sem, plot_widths, pretty_colorbar


def pretty_gat(scores, times=None, chance=0, ax=None, sig=None, cmap='RdBu_r',
               clim=None, colorbar=True, xlabel='Testing Times (s.)',
               ylabel='Train times (ms.)', sfreq=250):
    scores = np.array(scores)

    if times is None:
        times = np.arange(scores.shape[0]) / float(sfreq)

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
    im = ax.matshow(scores, extent=extent, cmap=cmap, origin='lower',
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
        pretty_colorbar(
            im, ax=ax, ticks=[vmin, chance, vmax],
            ticklabels=['%.2f' % vmin, 'Chance', '%.2f' % vmax])

    # setup ticks
    xticks, xticklabels = _set_ticks(test_times)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    yticks, yticklabels = _set_ticks(times)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    if len(xlabel):
        ax.set_xlabel(xlabel)
    if len(ylabel):
        ax.set_ylabel(ylabel)
    ax.set_xlim(min(test_times), max(test_times))
    ax.set_ylim(min(times), max(times))
    pretty_plot(ax)
    return ax


def pretty_decod(scores, times=None, chance=0, ax=None, sig=None, width=3.,
                 color='k', fill=False, xlabel='Times (ms.)', sfreq=250):
    scores = np.array(scores)

    if times is None:
        times = np.arange(scores.shape[0]) / float(sfreq)

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


def pretty_slices(scores, times=None, sig=None, sig_off=None, tois=None,
                   chance=0, axes=None, width=3., colors=['k', 'b'], sfreq=250):
    scores = np.array(scores)
    # Setup times
    if times is None:
        times = np.arange(scores.shape[0]) / float(sfreq)
    # Setup TOIs
    if tois is None:
        tois = np.linspace(min(times), max(times), 5)
    # Setup Figure
    if axes is None:
        fig, axes = plt.subplots(len(tois), 1, figsize=[5, 6])
    ymin = np.min(scores.mean(0) - scores.std(0)/np.sqrt(len(scores)))
    ymax = np.max(scores.mean(0) + scores.std(0)/np.sqrt(len(scores)))
    # Diagonalize
    scores_diag = np.array([np.diag(ii) for ii in scores])
    if sig is not None:
        sig = np.array(sig)
        sig_diag = np.diag(sig)
    for sel_time, ax in zip(tois, reversed(axes)):
        # Select TOI
        idx = np.argmin(abs(times - sel_time))
        scores_off = scores[:, idx, :] if sig is not None else None
        sig_off = sig[idx, :] if sig is not None else None
        if sig is not None:
            scores_sig = (scores_diag.mean(0) * (~sig_off[idx]) +
                          scores_off.mean(0) * (sig_off[idx]))
            ax.fill_between(times, scores_diag.mean(0), scores_sig,
                            color='yellow', alpha=.5, linewidth=0)
        pretty_decod(scores_off, times, chance, sig=sig_off,
                     width=width, color=colors[1], fill=False, ax=ax)
        pretty_decod(scores_diag, times, chance, sig=sig_diag,
                     width=width, color=colors[0], fill=False, ax=ax)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([ymin, chance, ymax])
        ax.set_yticklabels(['%.2f' % ymin, 'chance', '%.2f' % ymax])
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]], color='b',
                zorder=-2)
        # Add indicator
        ax.text(sel_time, ymin + .05 * np.ptp([ymin, ymax]),
                '%i ms.' % (np.array(sel_time) * 1e3),
                color='b', backgroundcolor='w', ha='center', zorder=-1)
        pretty_plot(ax)
        if ax != axes[-1]:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
    return axes
