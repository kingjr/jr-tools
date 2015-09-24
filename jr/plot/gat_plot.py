import matplotlib.pyplot as plt
import numpy as np
from .base import pretty_plot, plot_sem, plot_widths, pretty_colorbar


def pretty_gat(scores, times=None, chance=0, ax=None, sig=None, cmap='RdBu_r',
               clim=None, colorbar=True, xlabel='Test Times',
               ylabel='Train Times', sfreq=250, diagonal=None,
               test_times=None):
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
    if test_times is None:
        if scores.shape[1] == scores.shape[0]:
            test_times = times
        else:
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

    #
    if diagonal is not None:
        ax.plot([np.max([min(times), min(test_times)]),
                 np.min([max(times), max(test_times)])],
                [np.max([min(times), min(test_times)]),
                 np.min([max(times), max(test_times)])], color=diagonal)
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
                 color='k', fill=False, xlabel='Times', sfreq=250, alpha=.75):
    scores = np.array(scores)

    if (scores.ndim == 1) or (scores.shape[1] <= 1):
        scores = scores[:, None].T
    if times is None:
        times = np.arange(scores.shape[1]) / float(sfreq)

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
        if fill:
            scores_sig = (chance + (scores_m - chance) * sig)
            ax.fill_between(times, chance, scores_sig, color=color,
                            alpha=alpha, linewidth=0)
            ax.plot(times, scores_m, color='k')
            plot_widths(times, scores_m, widths, ax=ax, color='k')
        else:
            plot_widths(times, scores_m, widths, ax=ax, color=color)

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


def pretty_slices(scores, times=None, sig=None, sig_diagoff=None, tois=None,
                  chance=0, axes=None, width=3., colors=['k', 'b'], sfreq=250,
                  sig_invdiagoff=None, fill_color='yellow'):
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
    else:
        sig_diag = None
    for sel_time, ax in zip(tois, reversed(axes)):
        # Select TOI
        idx = np.argmin(abs(times - sel_time))
        scores_off = scores[:, idx, :]
        sig_off = sig[idx, :] if sig is not None else None
        if sig_diagoff is not None:
            scores_sig = (scores_diag.mean(0) * (~sig_diagoff[idx]) +
                          scores_off.mean(0) * (sig_diagoff[idx]))
            ax.fill_between(times, scores_diag.mean(0), scores_sig,
                            color=fill_color, alpha=.5, linewidth=0)
        if sig_invdiagoff is not None:
            scores_sig = (scores_diag.mean(0) * (~sig_invdiagoff[idx]) +
                          scores_off.mean(0) * (sig_invdiagoff[idx]))
            ax.fill_between(times, scores_diag.mean(0), scores_sig,
                            color='red', alpha=.5, linewidth=0)
        pretty_decod(scores_off, times, chance, sig=sig_off,
                     width=width, color=colors[1], fill=False, ax=ax)
        pretty_decod(scores_diag, times, chance, sig=sig_diag,
                     width=0, color=colors[0], fill=False, ax=ax)
        pretty_decod(scores_diag.mean(0), times, chance, sig=sig_diag,
                     width=width, color='k', fill=False, ax=ax)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([ymin, chance, ymax])
        ax.set_yticklabels(['%.2f' % ymin, 'chance', '%.2f' % ymax])
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]],
                color=colors[1], zorder=-2)
        # Add indicator
        ax.text(sel_time, ymin + .05 * np.ptp([ymin, ymax]),
                '%i ms' % (np.array(sel_time) * 1e3),
                color=colors[1], backgroundcolor='w', ha='center', zorder=-1)
        pretty_plot(ax)
        if ax != axes[-1]:
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
    return axes
