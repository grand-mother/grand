"""
Functions used to filter signal and change frame.

Functions initially written by Valentin Decoene and adapted to new formats.
"""

import numpy as np
from scipy.signal import butter, lfilter


def get_in_shower_plane_root(pos, k, core, Bx, By, Bz):
    """Transform in shower plane frame."""
    pos = (pos - core[:, np.newaxis])
    B = np.array([Bx, By, Bz])
    kxB = np.cross(k, B)
    kxB /= np.linalg.norm(kxB)
    kxkxB = np.cross(k, kxB)
    kxkxB /= np.linalg.norm(kxkxB)
    return np.array([np.dot(kxB, pos), np.dot(kxkxB, pos), np.dot(k, pos)])


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """Define subfunction of filter."""
    b, a = butter(5,
                  [lowcut / (0.5 * fs),
                   highcut / (0.5 * fs)],
                  btype='band')  # (order, [low, high], btype)

    return lfilter(b, a, data)


def filters_root(tin, vin, FREQMIN=50.e6, FREQMAX=200.e6):
    """Filter signal v(t) in given bandwidth.

    Parameters
    ----------
     : voltages
        The array of time (s) + voltage (muV) vectors to be filtered
     : FREQMIN
        The minimal frequency of the bandpass filter (Hz)
     : FREQMAX:
        The maximal frequency of the bandpass filter (Hz)

    Notes
    -----
    At present Butterworth filter only is implemented
    """
    fs = 1 / np.mean(np.diff(tin))  # Compute frequency step
    # print("Trace sampling frequency: ",fs/1e6,"MHz")
    nCh = np.shape(vin)[0]
    res = tin
    for i in range(nCh):
        vi = vin[i, :]
        res = np.append(res,
                        _butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs))

    res = np.reshape(res, (nCh+1, len(tin)))  # Put it back inright format
    return res
