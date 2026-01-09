import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample_poly


def _moving_average(x, window_len):
    if window_len <= 1:
        return x
    window_len = int(window_len)
    half = window_len // 2
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        out[i] = (csum[end] - csum[start]) / float(end - start)
    return out


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
    tmpSig = resample_poly(data, int(Fs), int(Fo))
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
