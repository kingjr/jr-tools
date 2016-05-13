import numpy as np
import matplotlib.pyplot as plt
from ..stats.circ import circ_shift, circ_double_tuning
from .base import plot_sem, pretty_plot
from ..utils import pi_labels


def plot_sem_polar(angles, radius, ax=None, color='b', linewidth=1, alpha=.5,
                   fill=True):
    if ax is None:
        ax = plt.subplot(111, polar=True)
    if radius.ndim == 1:
        radius = np.reshape(radius, [1, -1])
        radius = np.concatenate((radius, radius), axis=0)
    if len(angles) != radius.shape[1]:
        raise ValueError('angles and radius must have the same dimensionality')
    sem = np.std(radius, axis=0) / np.sqrt(len(angles))
    m = np.mean(radius, axis=0)
    ax.plot(angles, m, color=color, linewidth=linewidth)
    ax.fill_between(np.hstack((angles, angles[::-1])),
                    np.hstack((m, m[::-1])) + np.hstack((sem, -sem[::1])),
                    facecolor=color, edgecolor='none', alpha=alpha)
    if fill:
        ax.fill_between(angles, m, facecolor=color, edgecolor='none',
                        alpha=alpha)
    return ax


def pretty_polar_plot(ax):
    ax.set_rticks([])
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    ax.spines['polar'].set_color('dimgray')
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5)[:-1])
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$-\pi/2$'])


def plot_tuning(data, shift=0, half=False, polar=False, ax=None, chance='auto',
                color='k', alpha=.5, ylim='auto'):
    data = np.array(data)
    if data.ndim == 1:
        data = np.reshape(data, [1, -1])
    nbin = data.shape[1]
    bins = np.linspace(0, 2 * np.pi, nbin)
    if shift:
        data = circ_shift(data.T, shift, wrapped=False).T
        bins = bins - shift
    if chance == 'auto':
        chance = 1. / nbin
    if ylim == 'auto':
        ylim = [min(data.mean(0) - data.std(0) / np.sqrt(len(data))),
                max(data.mean(0) + data.std(0) / np.sqrt(len(data)))]
    if polar:
        bins = np.hstack((bins, bins[0]))
        data = np.hstack((data, data[:, 0, None]))
        if half:
            data, bins = circ_double_tuning(data.T, bins)
            data = data.T
        ax = plot_sem_polar(bins, data, color=color, alpha=alpha, fill=False,
                            ax=ax)
        if chance is not None:
            ax.fill_between(
                np.hstack((bins, bins[-1::-1])),
                np.hstack((chance * np.ones(len(bins)), data.mean(0)[-1::-1])),
                facecolor=color, edgecolor='none', alpha=alpha)
        pretty_polar_plot(ax)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(pi_labels([-np.pi, 0, np.pi]))
    else:
        ax = plot_sem(bins, data, ax=ax, color=color, alpha=alpha)
        if chance is not None:
            ax.fill_between(
                np.hstack((bins[0], bins, bins[-1], bins[0])),
                np.hstack((chance, data.mean(0), chance, chance)),
                facecolor=color, edgecolor='none', alpha=alpha)
        ax.set_xlim(bins[0], bins[-1])
        pretty_plot(ax)
        ax.set_xticks(np.linspace(bins[0], bins[-1], 3))
        ax.set_xticklabels(pi_labels(np.linspace(bins[0], bins[-1], 3)))
        ax.set_xlabel('Angle')
        ax.set_ylim(ylim[0], ylim[1])
        if chance is not None:
            ax.set_yticks([ylim[0], chance, ylim[1]])
            ax.set_yticklabels(['%.2f' % (100 * ii)
                                for ii in [ylim[0], chance, ylim[1]]])
        else:
            ax.set_yticks([ylim[0], ylim[1]])
            ax.set_yticklabels(['%.2f' % (100 * ii) for ii in ylim])
    return ax


def circ_fill_between(data0, data1, ax=None, color='k', alpha=1., shift=False,
                      half=False):
    if ax is None:
        ax = plt.gca()

    def circ(data):
        data = np.array(data)
        nbin = len(data)
        if shift:
            data = circ_shift(data, shift, wrapped=False)
        bins = np.linspace(0, 2 * np.pi, nbin)
        bins = np.hstack((bins, bins[0]))
        data = np.hstack((data, data[0, None]))
        if half:
            data, bins = circ_double_tuning(data, bins)
        return data, bins

    data0, bins = circ(data0)
    data1, bins = circ(data1)

    ax.fill_between(
        np.hstack((bins, bins[-1::-1])),
        np.hstack((data0, data1[-1::-1])), color=color, alpha=alpha)
    return ax
