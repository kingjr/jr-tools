import matplotlib.pyplot as plt
import numpy as np
from . import pretty_plot


def plot_butterfly(evoked, ax=None, sig=None, color=None, ch_type=None):
    from mne import pick_types
    if ch_type is not None:
        picks = pick_types(evoked.info, ch_type)
        evoked = evoked.copy()
        evoked = evoked.pick_types(ch_type)
        sig = sig[picks, :] if sig is not None else None
    times = evoked.times * 1e3
    data = evoked.data
    ax = plt.gca() if ax is None else ax
    ax.plot(times, data.T, color='k', alpha=.5)
    gfp = np.vstack((data.max(0), data.min(0)))
    if sig is not None:
        sig = np.array(np.sum(sig, axis=0) > 0., dtype=int)
        ax.fill_between(np.hstack((times, times[::-1])),
                        np.hstack((sig * gfp[0, :] + (1 - sig) * gfp[1, :],
                                   gfp[1, ::-1])),
                        facecolor=color, edgecolor='none', alpha=.5,
                        zorder=len(data) + 1)
    ax.axvline(0, color='k')
    ax.set_xlabel('Times (ms)')
    ax.set_xlim(min(times), max(times))
    xticks = np.arange(np.ceil(min(times)/1e2) * 1e2,
                       np.floor(max(times)/1e2) * 1e2 + 1e-10, 100)
    ax.set_xticks(xticks)
    print xticks
    ax.set_xticklabels(['%i' % t if t in [xticks[0], xticks[-1], 0]
                        else '' for t in xticks])
    ax.set_yticks([np.min(data), np.max(data)])
    ax.set_ylim(np.min(data), np.max(data))
    ax.set_xlim(np.min(times), np.max(times))
    pretty_plot(ax)
    return ax


def plot_gfp(evoked, ax=None, sig=None, color=None, ch_type='mag'):
    from mne import pick_types
    if ch_type is not None:
        picks = pick_types(evoked.info, ch_type)
        evoked = evoked.copy()
        evoked = evoked.pick_types(ch_type)
        sig = sig[picks, :] if sig is not None else None
    times = evoked.times * 1e3
    gfp = np.std(evoked.data, axis=0)
    ax = plt.gca() if ax is None else ax
    ax.plot(times, gfp, color='k', alpha=.5)
    if sig is not None:
        sig = np.array(np.sum(sig, axis=0) > 0., dtype=int)
        ax.fill_between(np.hstack((times, times[::-1])),
                        np.hstack((sig * gfp, np.zeros_like(gfp))),
                        facecolor=color, edgecolor='none', alpha=.5)
    ax.axvline(0, color='k')
    ax.set_xlabel('Times (ms)')
    ax.set_xlim(min(times), max(times))
    xticks = np.arange(np.ceil(min(times)/1e2) * 1e2,
                       np.floor(max(times)/1e2) * 1e2 + 1e-10, 100)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%i' % t if t in [xticks[0], xticks[-1], 0]
                        else '' for t in xticks])
    ax.set_yticks([np.min(gfp), np.max(gfp)])
    ax.set_ylim(np.min(gfp), np.max(gfp))
    ax.set_xlim(np.min(times), np.max(times))
    pretty_plot(ax)
    return ax
