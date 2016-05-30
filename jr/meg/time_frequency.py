import numpy as np
from mne.utils import logger
from mne.time_frequency.tfr import _check_decim, morlet, cwt
from mne.parallel import parallel_func


def single_trial_tfr(data, sfreq, frequencies, use_fft=True, n_cycles=7,
                     decim=1, n_jobs=1, zero_mean=False, verbose=None):
    """Compute time-frequency decomposition single epochs.

    Parameters
    ----------
    data : array of shape [n_epochs, n_channels, n_times]
        The epochs
    sfreq : float
        Sampling rate
    frequencies : array-like
        The frequencies
    use_fft : bool
        Use the FFT for convolutions or not.
    n_cycles : float | array of float
        Number of cycles  in the Morlet wavelet. Fixed number
        or one per frequency.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice` returns tfr[..., decim].
        Note that decimation may create aliasing artifacts.
        Defaults to 1.
    n_jobs : int
        The number of epochs to process at the same time
    zero_mean : bool
        Make sure the wavelets are zero mean.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    tfr : 4D array, shape (n_epochs, n_chan, n_freq, n_time)
        Time frequency estimate (complex).
    """
    decim = _check_decim(decim)
    mode = 'same'
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data[:, :, decim].shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(sfreq, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    parallel, my_cwt, _ = parallel_func(cwt, n_jobs)

    logger.info("Computing time-frequency deomposition on single epochs...")

    out = np.empty((n_epochs, n_channels, n_frequencies, n_times),
                   dtype=np.complex128)

    # Package arguments for `cwt` here to minimize omissions where only one of
    # the two calls below is updated with new function arguments.
    cwt_kw = dict(Ws=Ws, use_fft=use_fft, mode=mode, decim=decim)
    if n_jobs == 1:
        for k, e in enumerate(data):
            out[k] = cwt(e, **cwt_kw)
    else:
        # Precompute tf decompositions in parallel
        tfrs = parallel(my_cwt(e, **cwt_kw) for e in data)
        for k, tfr in enumerate(tfrs):
            out[k] = tfr

    return out
