import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample


def _moving_average(x, window_len):
    if window_len <= 1:
        return x
    kernel = np.ones(int(window_len), dtype=float) / float(window_len)
    return np.convolve(x, kernel, mode="same")


def detectIEDs(data, Fo):
    """
    DETECTIEDS finds interictal epileptiform discharges (IEDs).

    Args:
        data: numpy array shaped (samples,) for a single channel
        Fo: original sampling rate

    Returns:
        spike_count: integer count of detected spikes.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("detectIEDs expects a single-channel 1D array.")
    n_samps = data.shape[0]

    # detection parameters
    Fs = 400
    detection_threshold = 8
    window_size = 1

    # filtering and downsampling
    b, a = butter(4, [20 / (Fs / 2), 40 / (Fs / 2)], btype="band")
    n_resamp = int(np.floor(n_samps / (Fo / Fs)))
    tmpSig = resample(data, n_resamp)
    tmpSig = tmpSig - np.mean(tmpSig)
    data2040 = filtfilt(b, a, tmpSig)

    snr = detection_threshold * np.std(np.abs(data2040))
    smoothed = _moving_average(np.abs(data2040), int(Fs / 50))
    peaks, _ = find_peaks(smoothed, height=snr)

    locs_1based = peaks + 1

    # retain those detections occurring within 250 ms
    if len(locs_1based):
        diffs = np.diff(locs_1based, prepend=0)
        loc_inds = ~(diffs < (window_size * Fs / 4))
        chosen_locs_1based = locs_1based[loc_inds]
    else:
        chosen_locs_1based = np.array([], dtype=int)

    return int(len(chosen_locs_1based))
