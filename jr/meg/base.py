import numpy as np


def mat2mne(data, chan_names='meg', chan_types=None, sfreq=250, events=None,
            tmin=0):
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
                       np.zeros(n_trial, int), np.zeros(n_trial)]
    else:
        events = np.array(events, int)
        if events.ndim == 1:
            events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                           np.zeros(n_trial), events]
        elif (events.ndim != 2) or (events.shape[1] != 3):
            raise ValueError('events shape must be ntrial, or ntrials * 3')

    info = create_info(chan_names, sfreq, chan_types)
    return EpochsArray(data, info, events=np.array(events, int), verbose=False,
                       tmin=tmin)


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
    events = np.array(np.round(events), int)

    # transform into epochs
    new_epochs = EpochsArray(meta_data, epochs.info, events=events,
                             verbose=False)
    new_epochs.events = np.array(new_epochs.events, float)
    new_epochs.events[:, 2] = meta_y

    # XXX why change time and sfreq?
    new_epochs.times = epochs.times
    new_epochs.info['sfreq'] = epochs.info['sfreq']
    return new_epochs


def resample_epochs(epochs, sfreq):
    """Fast MNE epochs resampling"""
    # from librosa import resample
    # librosa.resample(channel, o_sfreq, sfreq, res_type=res_type)
    from scipy.signal import resample

    # resample
    epochs._data = resample(
        epochs._data, epochs._data.shape[2] / epochs.info['sfreq'] * sfreq,
        axis=2)
    # update metadata
    epochs.info['sfreq'] = sfreq
    epochs.times = (np.arange(epochs._data.shape[2],
                              dtype=np.float) / sfreq + epochs.times[0])
    return epochs


def detect_bad_channels(inst, pick_types=None, threshold=.2):
    from sklearn.preprocessing import RobustScaler
    from sklearn.covariance import EmpiricalCovariance
    from jr.stats import median_abs_deviation
    if pick_types is None:
        pick_types = dict(meg='mag')
    inst = inst.pick_types(copy=True, **pick_types)
    cov = EmpiricalCovariance()
    cov.fit(inst._data.T)
    cov = cov.covariance_
    # center
    scaler = RobustScaler()
    cov = scaler.fit_transform(cov).T
    cov /= median_abs_deviation(cov)
    cov -= np.median(cov)
    # compute robust summary metrics
    mu = np.median(cov, axis=0)
    sigma = median_abs_deviation(cov, axis=0)
    mu /= median_abs_deviation(mu)
    sigma /= median_abs_deviation(sigma)
    distance = np.sqrt(mu ** 2 + sigma ** 2)
    bad = np.where(distance < threshold)[0]
    bad = [inst.ch_names[ch] for ch in bad]
    return bad


def forward_pipeline(raw_fname, freesurfer_dir, subject,
                     trans_fname=None, fwd_fname=None, oct_fname=None,
                     bem_sol_fname=None, save_dir=None, overwrite=False):
    import os.path as op
    from jr.meg import check_freesurfer, mne_anatomy

    # Setup paths
    if save_dir is None:
        save_dir = '/'.join(raw_fname.split('/')[:-1])
        print('Save/read directory: %s' % save_dir)

    if trans_fname is None:
        trans_fname = op.join(save_dir, subject + '-trans.fif')
        bem_sol_fname = op.join(freesurfer_dir, subject, 'bem',
                                subject + '-5120-bem-sol.fif')
        oct_fname = op.join(freesurfer_dir, subject, 'bem',
                            subject + '-oct-6-src.fif')
        fwd_fname = op.join(save_dir, subject + '-meg-fwd.fif')

    # Checks Freesurfer segmentation and compute watershed bem
    if check_freesurfer(subjects_dir=freesurfer_dir,
                        subject=subject):
        mne_anatomy(subjects_dir=freesurfer_dir, subject=subject,
                    overwrite=overwrite)

    # Manual coregisteration head markers with coils
    if overwrite or not op.isfile(trans_fname):
        from mne.gui import coregistration
        coregistration(subject=subject, subjects_dir=freesurfer_dir,
                       inst=raw_fname)

    # Forward solution
    if overwrite or not op.exists(fwd_fname):
        from mne import (make_forward_solution, convert_forward_solution,
                         write_forward_solution)
        fwd = make_forward_solution(
            raw_fname, trans_fname, oct_fname, bem_sol_fname,
            fname=None, meg=True, eeg=False, mindist=5.0,
            overwrite=True, ignore_ref=True)

        # convert to surface orientation for better visualization
        fwd = convert_forward_solution(fwd, surf_ori=True)
        # save
        write_forward_solution(fwd_fname, fwd, overwrite=True)
    else:
        from mne import read_forward_solution
        fwd = read_forward_solution(fwd_fname, surf_ori=True)
    return fwd


def add_channels(inst, data, ch_names, ch_types):
    from mne.io import _BaseRaw, RawArray
    from mne.epochs import _BaseEpochs, EpochsArray
    from mne import create_info
    if 'meg' in ch_types or 'eeg' in ch_types:
        return NotImplementedError('Can only add misc, stim and ieeg channels')
    info = create_info(ch_names=ch_names, sfreq=inst.info['sfreq'],
                       ch_types=ch_types)
    if isinstance(inst, _BaseRaw):
        for key in ('buffer_size_sec', 'filename'):
            info[key] = inst.info[key]
        new_inst = RawArray(data, info=info, first_samp=inst._first_samps[0])
    elif isinstance(inst, _BaseEpochs):
        new_inst = EpochsArray(data, info=info)
    else:
        raise ValueError('unknown inst type')
    return inst.add_channels([new_inst], copy=True)


def decimate(inst, decim, copy=False):
    """Decimate"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    inst = inst.copy() if copy else inst
    if isinstance(inst, _BaseRaw):
        inst._data = inst._data[:, ::decim]
        inst.info['sfreq'] //= decim
        inst._first_samps //= decim
        inst.first_samp //= decim
        inst._last_samps //= decim
        inst.last_samp //= decim
        inst._raw_lengths //= decim
        inst._times = inst._times[::decim]
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:, :, ::decim]
        inst.info['sfreq'] //= decim
        inst.times = inst.times[::decim]
    return inst
