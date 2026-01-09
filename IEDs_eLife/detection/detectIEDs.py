import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, peak_widths, resample


def _update_user(*_args, **_kwargs):
    # Placeholder for MATLAB's updateUser; keep silent in Python.
    return None


def _moving_average(x, window_len):
    if window_len <= 1:
        return x
    kernel = np.ones(int(window_len), dtype=float) / float(window_len)
    return np.convolve(x, kernel, mode="same")


def detectIEDs(data, Fo):
    """
    DETECTIEDS finds interictal epileptiform discharges (IEDs).

    Args:
        data: numpy array shaped (channels, samples)
        Fo: original sampling rate

    Returns:
        IEDdata: dict mirroring the MATLAB structure.
    """
    data = np.asarray(data)
    n_chans, n_samps = data.shape

    IEDdata = {}
    IEDdata["parameters"] = {}
    IEDdata["parameters"]["OGdataLength"] = max(data.shape)
    IEDdata["parameters"]["OGsamplingRate"] = Fo

    # detection parameters
    Fs = 400
    IEDdata["parameters"]["downSamplingRate"] = Fs
    IEDdata["parameters"]["detectionThreshold"] = 8
    IEDdata["parameters"]["artifactThreshold"] = 20
    IEDdata["parameters"]["windowSize"] = 1

    # filtering and downsampling
    b, a = butter(4, [20 / (Fs / 2), 40 / (Fs / 2)], btype="band")
    n_resamp = int(np.floor(n_samps / (Fo / Fs)))
    IEDmat = np.zeros((n_chans, n_resamp), dtype=bool)
    tSec = np.linspace(0, n_samps / Fs, n_resamp)

    IEDdata["measurements"] = []
    IEDdata["foundPeaks"] = []
    IEDdata["detections"] = []
    IEDdata["fullBWdata"] = [[] for _ in range(n_chans)]
    IEDdata["resampledData"] = [[] for _ in range(n_chans)]

    for ch in range(n_chans):
        _update_user(ch + 1, 10, n_chans, "downsampling and filtering channel")

        tmpSig = resample(data[ch, :], n_resamp)
        tmpSig = tmpSig - np.mean(tmpSig)
        data2040 = filtfilt(b, a, tmpSig)

        IEDdata["resampledDataLength"] = len(tmpSig)

        snr = IEDdata["parameters"]["detectionThreshold"] * np.std(np.abs(data2040))
        art_thresh = IEDdata["parameters"]["artifactThreshold"] * np.std(np.abs(data2040))
        IEDdata["measurements"].append({"SNR": snr, "ArtifactThresh": art_thresh})

        smoothed = _moving_average(np.abs(data2040), int(Fs / 50))
        peaks, props = find_peaks(smoothed, height=snr)

        prominences = peak_prominences(smoothed, peaks)[0] if len(peaks) else np.array([])
        widths = peak_widths(smoothed, peaks)[0] if len(peaks) else np.array([])

        locs_1based = peaks + 1
        IEDdata["foundPeaks"].append(
            {
                "peaks": smoothed[peaks] if len(peaks) else np.array([]),
                "locs": locs_1based,
                "peakWidth": widths,
                "peakProminence": prominences,
            }
        )

        # retain those detections occurring within 250 ms
        if len(locs_1based):
            diffs = np.diff(locs_1based, prepend=0)
            loc_inds = ~(diffs < (IEDdata["parameters"]["windowSize"] * Fs / 4))
            chosen_locs_1based = locs_1based[loc_inds]
        else:
            chosen_locs_1based = np.array([], dtype=int)

        IEDdata["detections"].append({"times": chosen_locs_1based})

        n_peaks = len(chosen_locs_1based)
        for pk in range(n_peaks):
            _update_user("extracting peak", pk + 1, 10, n_peaks)

            center_resamp = chosen_locs_1based[pk] - 1
            half_resamp_floor = int(np.floor(IEDdata["parameters"]["windowSize"] * Fs / 2))
            half_resamp_ceil = int(np.ceil(IEDdata["parameters"]["windowSize"] * Fs / 2))

            if (center_resamp - half_resamp_floor) > 0 and (
                center_resamp + half_resamp_ceil
            ) < len(tmpSig):
                center_orig_1based = int(round((Fo / Fs) * chosen_locs_1based[pk]))
                half_orig_floor = int(np.floor(IEDdata["parameters"]["windowSize"] * Fo / 2))
                half_orig_ceil = int(np.ceil(IEDdata["parameters"]["windowSize"] * Fo / 2))

                start_orig = center_orig_1based - 1 - half_orig_floor
                end_orig = center_orig_1based - 1 + half_orig_ceil
                start_resamp = center_resamp - half_resamp_floor
                end_resamp = center_resamp + half_resamp_ceil

                IEDdata["fullBWdata"][ch].append(
                    {
                        "windowedData": data[
                            :, start_orig : end_orig + 1
                        ]
                    }
                )
                IEDdata["resampledData"][ch].append(
                    {
                        "windowedData": tmpSig[
                            start_resamp : end_resamp + 1
                        ]
                    }
                )

            IEDmat[ch, center_resamp] = True

            IEDdata["tSamps"] = np.arange(
                chosen_locs_1based[pk]
                - int(np.floor(IEDdata["parameters"]["windowSize"] * Fo / 2)),
                chosen_locs_1based[pk]
                + int(np.ceil(IEDdata["parameters"]["windowSize"] * Fo / 2))
                + 1,
            )
            IEDdata["tReSamps"] = np.arange(
                chosen_locs_1based[pk]
                - int(np.floor(IEDdata["parameters"]["windowSize"] * Fs / 2)),
                chosen_locs_1based[pk]
                + int(np.ceil(IEDdata["parameters"]["windowSize"] * Fs / 2))
                + 1,
            )

    IEDdata["IEDmat"] = IEDmat
    IEDdata["tSec"] = tSec

    return IEDdata
