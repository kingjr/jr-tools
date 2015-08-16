import numpy as np


def mat2mne(data, chan_names='meg', chan_types=None, sfreq=250, events=None):
    from mne.epochs import EpochsArray
    from mne.io.meas_info import create_info
    data = np.array(data)
    n_trial, n_chan, n_time = data.shape
    # chan names
    if isinstance(chan_names, str):
        chan_names = [chan_names + '_%02i' % chan for chan in range(n_chan)]
    if len(chan_names) != n_chan:
        raise ValueError('chan_names must be a string or a list of'
                         'n_chan strings')
    # chan types
    if isinstance(chan_types, str):
        chan_types = [chan_types] * n_chan
    elif chan_types is None:
        if isinstance(chan_names, str):
            if chan_names != 'meg':
                chan_types = [chan_names] * n_chan
            else:
                chan_types = ['mag'] * n_chan
        elif isinstance(chan_names, list):
            chan_types = ['mag' for chan in chan_names]
        else:
            raise ValueError('Specify chan_types')

    # events
    if events is None:
        events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                       np.zeros(n_trial), np.zeros(n_trial)]
    else:
        events = np.array(events)
        if events.ndim == 1:
            events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                           np.zeros(n_trial), events]
        elif (events.ndim != 2) or (events.shape[1] != 3):
            raise ValueError('events shape must be ntrial, or ntrials * 3')

    info = create_info(chan_names, sfreq, chan_types)
    return EpochsArray(data, info, events=events, verbose=False)


def make_meta_epochs(epochs, y, n_bin=100):
    from scipy.stats import trim_mean
    from mne.epochs import EpochsArray
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

    events = np.vstack((np.zeros(len(meta_y)),
                        np.zeros(len(meta_y)), meta_y)).T
    events = np.round(events)

    # transform into epochs
    new_epochs = EpochsArray(meta_data, epochs.info, events=events,
                             verbose=False)
    new_epochs.events[:, 2] = meta_y

    # XXX why change time and sfreq?
    new_epochs.times = epochs.times
    new_epochs.info['sfreq'] = epochs.info['sfreq']
    return new_epochs
