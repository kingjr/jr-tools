import numpy as np


def make_meta_epochs(epochs, y, n_bin=100):
    from scipy.stats import trim_mean
    meta_data = list()  # EEG data
    meta_y = list()  # regressors
    n = len(epochs)

    # make continuous y into bins to become categorical
    if len(np.unique(y)) < n_bin:
        hist, bin_edge = np.histogram(y, n_bin)
        y_ = y
        for low, high in zip(bin_edge[:-1], bin_edge[1:]):
            sel = np.where((y >= low) & (y < high))[0]
            y_[sel] = .5 * (high + low)
        y = y_

    # if discrete and few categories
    if len(np.unique(y)) < n_bin:
        already_used = list()
        for this_y in np.unique(y):
            for ii in range(n / len(np.unique(y)) / n_bin):
                sel = np.where(y == this_y)[0]
                sel = [ii for ii in sel if ii not in already_used][:n_bin]
                if not len(sel):
                    continue
                meta_data.append(trim_mean(epochs._data[sel, :, :], .05,
                                           axis=0))
                meta_y.append(this_y)
                already_used += sel
    else:
        hist, bin_edge = np.histogram(y, n_bin)
        for low, high in zip(bin_edge[:-1], bin_edge[1:]):
            sel = np.where((y >= low) & (y < high))[0]
            this_y = .5 * (high + low)
            if not len(sel):
                continue
            meta_data.append(trim_mean(epochs._data[sel, :, :], .05, axis=0))
            meta_y.append(this_y)

    # transform into epochs
    n = len(meta_y)
    epochs._data = np.array(meta_data)
    epochs.events = np.vstack((np.zeros(n), np.zeros(n), meta_y)).T
    return epochs
