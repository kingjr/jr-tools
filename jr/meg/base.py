import numpy as np
import os
import os.path as op
import warnings


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


def check_freesurfer(subjects_dir, subject):
    # Check freesurfer finished without any errors
    fname = op.join(subjects_dir, subject, 'scripts', 'recon-all.log')
    if op.isfile(fname):
        with open(fname, 'rb') as fh:
            fh.seek(-1024, 2)
            last = fh.readlines()[-1].decode()
        print(last)
        print('{}: ok'.format(subject))
        return True
    else:
        print('{}: missing'.format(subject))
        return False


def check_libraries():
    """Raise explicit error if mne and freesurfer or mne c are not installed"""
    from mne.utils import has_mne_c, has_freesurfer
    import subprocess
    if not (has_freesurfer() and has_mne_c() and
            op.isfile(subprocess.check_output(['which', 'freesurfer'])[:-1])):
        # export FREESURFER_HOME=/usr/local/freesurfer
        # source $FREESURFER_HOME/SetUpFreeSurfer.sh
        # export MNE_ROOT=/home/jrking/MNE-2.7.4-3452-Linux-x86_64
        # source $MNE_ROOT/bin/mne_setup_sh
        # export LD_LIBRARY_PATH=/home/jrking/anaconda/lib/
        raise('Check your freesurfer and mne c paths')


def anatomy_pipeline(subject, subjects_dir=None, overwrite=False):
    from mne.bem import make_watershed_bem
    from mne.commands.mne_make_scalp_surfaces import _run as make_scalp_surface
    from mne.utils import get_config
    if subjects_dir is None:
        subjects_dir = get_config('SUBJECTS_DIR')

    # Set file name ----------------------------------------------------------
    bem_dir = op.join(subjects_dir, subject, 'bem')
    src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')
    bem_fname = op.join(bem_dir, subject + '-5120-bem.fif')
    bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')

    # 0. Create watershed BEM surfaces
    if overwrite or not op.isfile(op.join(bem_dir, subject + '-head.fif')):
        check_libraries()
        if not check_freesurfer(subjects_dir=subjects_dir, subject=subject):
            warnings.warn('%s is probably not segmented correctly, check '
                          'log.' % subject)
        make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                           overwrite=True, volume='T1', atlas=False,
                           gcaatlas=False, preflood=None)

    # 1. Make scalp surfaces
    miss_surface = False
    # make_scalp is only for outer_skin
    for part in ['brain', 'inner_skull', 'outer_skin', 'outer_skull']:
        fname = op.join(
            bem_dir, 'watershed', '%s_%s_surface' % (subject, part))
        if not op.isfile(fname):
            miss_surface = True
    if overwrite or miss_surface:
        make_scalp_surface(subjects_dir=subjects_dir, subject=subject,
                           force=True, overwrite=True, verbose=None)

    # 2. Copy files outside watershed folder in case of bad manipulation
    miss_surface_copy = False
    for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
        fname = op.join(bem_dir, '%s.surf' % surface)
        if not op.isfile(fname):
            miss_surface_copy = True
    if overwrite or miss_surface_copy:
        for surface in ['inner_skull', 'outer_skull', 'outer_skin']:
            from shutil import copyfile
            from_file = op.join(bem_dir,
                                'watershed/%s_%s_surface' % (subject, surface))
            to_file = op.join(bem_dir, '%s.surf' % surface)
            if op.exists(to_file):
                os.remove(to_file)
            copyfile(from_file, to_file)

    # 3. Setup source space
    if overwrite or not op.isfile(src_fname):
        from mne import setup_source_space, write_source_spaces
        check_libraries()
        files = ['lh.white', 'rh.white', 'lh.sphere', 'rh.sphere']
        for fname in files:
            if not op.exists(op.join(subjects_dir, subject, 'surf', fname)):
                raise RuntimeError('missing: %s' % fname)

        src = setup_source_space(subject=subject, subjects_dir=subjects_dir,
                                 spacing='oct6', surface='white',
                                 add_dist=True, n_jobs=-1, verbose=None)
        write_source_spaces(src_fname, src)

    # 4. Prepare BEM model
    if overwrite or not op.exists(bem_sol_fname):
        from mne.bem import (make_bem_model, write_bem_surfaces,
                             make_bem_solution, write_bem_solution)
        check_libraries()
        surfs = make_bem_model(subject=subject, subjects_dir=subjects_dir)
        write_bem_surfaces(fname=bem_fname, surfs=surfs)
        bem = make_bem_solution(surfs)
        write_bem_solution(fname=bem_sol_fname, bem=bem)


def forward_pipeline(raw_fname, subject, fwd_fname=None, trans_fname=None,
                     subjects_dir=None, overwrite=False, ignore_ref=True):
    import os.path as op
    from mne.utils import get_config
    if subjects_dir is None:
        subjects_dir = get_config('SUBJECTS_DIR')

    # Setup paths
    save_dir = raw_fname.split('/')
    save_dir = ('/'.join(save_dir[:-1])
                if isinstance(save_dir, list) else save_dir)
    bem_dir = op.join(subjects_dir, subject, 'bem')

    bem_sol_fname = op.join(subjects_dir, subject, 'bem',
                            subject + '-5120-bem-sol.fif')
    oct_fname = op.join(subjects_dir, subject, 'bem',
                        subject + '-oct-6-src.fif')
    src_fname = op.join(bem_dir, subject + '-oct-6-src.fif')
    bem_sol_fname = op.join(bem_dir, subject + '-5120-bem-sol.fif')
    if trans_fname is None:
        trans_fname = op.join(save_dir, subject + '-trans.fif')
    if fwd_fname is None:
        fwd_fname = op.join(save_dir, subject + '-meg-fwd.fif')

    # 0. Checks Freesurfer segmentation and compute watershed bem
    miss_anatomy = not op.isfile(src_fname) or not op.exists(bem_sol_fname)
    for fname in [bem_sol_fname, oct_fname]:
        if not op.isfile(op.join(subjects_dir, subject, 'bem', fname)):
            miss_anatomy = True
    if miss_anatomy:
        raise RuntimeError('Could not find BEM (%s, %s), relaunch '
                           'pipeline_anatomy()' % (bem_sol_fname, oct_fname))

    # 1. Manual coregisteration head markers with coils
    if not op.isfile(trans_fname):
        raise RuntimeError('Could not find trans (%s), launch'
                           'coregistration.' % trans_fname)

    # 2. Forward solution
    if overwrite or not op.isfile(fwd_fname):
        from mne import (make_forward_solution, convert_forward_solution,
                         write_forward_solution)
        fwd = make_forward_solution(
            info=raw_fname, trans=trans_fname, src=oct_fname,
            bem=bem_sol_fname, meg=True, eeg=False, mindist=5.0,
            ignore_ref=ignore_ref)

        # Convert to surface orientation for better visualization
        fwd = convert_forward_solution(fwd, surf_ori=True)
        # save
        write_forward_solution(fwd_fname, fwd, overwrite=True)
    return


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
        inst._last_samps //= decim
        inst._raw_lengths[0] //= decim  # XXX why [0]? doesn't work
        inst._times = inst._times[::decim]
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:, :, ::decim]
        inst.info['sfreq'] //= decim
        inst.times = inst.times[::decim]
    return inst


def anonymize(info):
    """to anonymize epochs"""
    if info.get('subject_info') is not None:
        del info['subject_info']
    info['meas_date'] = [0, 0]
    for key_1 in ('file_id', 'meas_id'):
        for key_2 in ('secs', 'msecs', 'usecs'):
            info[key_1][key_2] = 0
