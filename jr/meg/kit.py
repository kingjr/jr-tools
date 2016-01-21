import numpy as np


def least_square_reference(inst, empty_room=None, max_times_samples=2000,
                           bad_channels=None, scaler=None, mrk=None,
                           elp=None, hsp=None):
    """
    Fits and applies Least Square projection of the reference channels
    (potentially from an empty room) and removes the corresponding component
    from the recordings of a subject.

    Parameters
    ----------
        inst : Raw | str
            Raw instance or path to raw data.
        empty_room : str | None
            Path to raw data acquired in empty room.
        max_times_samples : int
            Number of time sample to use for pinv. Defautls to 2000
        bad_channels : list | array, shape (n_chans) of strings
            Lists bad channels
        scaler : function | None
            Scaler functions to normalize data. Defaults to
            sklearn.preprocessing.RobustScaler.

    Returns
    -------
        inst : Raw

    adapted from Adeen Flinker 6/2013 (<adeen.f@gmail.com>) LSdenoise.m

    Main EHN
        - Automatically detects channel types.
        - Allows flexible scaler; Robust by default.
        - The data is projected back in Tesla.
        - Allows memory control.
    TODO:
        - Allow other kind of MNE-Python inst
        - Allow baseline selection (pre-stim instead of empty room)
        - Clean up memory
        - Allow fancy solver (l1, etc)
    """
    from scipy.linalg import pinv
    from mne.io import read_raw_kit
    from mne.io import _BaseRaw

    # Least square can be fitted on empty room or on subject's data
    if empty_room is None:
        if not isinstance(inst, _BaseRaw):
            raw = read_raw_kit(inst, preload=True)
        else:
            raw = inst
    else:
        if not isinstance(empty_room, _BaseRaw):
            raw = read_raw_kit(empty_room, preload=True)
        else:
            raw = empty_room

    # Parameters
    n_chans, n_times = raw._data.shape
    chan_info = raw.info['chs']

    # KIT: axial gradiometers (equiv to mag)
    ch_mag = np.where([ch['coil_type'] == 6001 for ch in chan_info])[0]
    # KIT: ref magnetometer
    ch_ref = np.where([ch['coil_type'] == 6002 for ch in chan_info])[0]
    # Other channels
    ch_misc = np.where([ch['coil_type'] not in [6001, 6002]
                        for ch in chan_info])[0]
    # Bad channel
    ch_bad = np.empty(0)
    if (bad_channels is not None) and len(bad_channels):
        bad_channels = [ii for ii, ch in enumerate(raw.ch_names)
                        if ch in bad_channels]
        bad_channels = np.array(bad_channels, int)
    # To avoid memory error, let's subsample across time
    sel_times = slice(0, n_times, int(np.ceil(n_times // max_times_samples)))

    # Whiten data
    if scaler is None:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    data_bsl = scaler.fit_transform(raw._data.T)

    # Fit Least Square coefficients on baseline data
    empty_sensors = data_bsl[:, ch_mag]
    if len(ch_bad):
        empty_sensors[:, ch_bad] = 0  # remove bad channels
    coefs = np.dot(pinv(data_bsl[sel_times, ch_ref]),
                   empty_sensors[sel_times, :])
    empty_sensors, data_bsl = None, None  # clear memory

    # Apply correction on subject data
    if empty_room is not None:
        del raw
        raw = read_raw_kit(inst, preload=True)

    data_subject = scaler.transform(raw._data.T)
    subject_sensors = (data_subject[:, ch_mag] -
                       np.dot(data_subject[:, ch_ref], coefs))

    # Remove bad channels
    if len(ch_bad):
        subject_sensors[:, ch_bad] = 0

    # Reproject baseline
    new_ref = np.dot(subject_sensors, pinv(coefs))

    # Un-whiten data to get physical units back
    data = np.concatenate((subject_sensors, new_ref,
                           raw._data[ch_misc, :].T), axis=1)
    data = scaler.inverse_transform(data)

    # Output
    raw._data = data.T
    return raw
