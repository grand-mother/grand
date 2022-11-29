"""! Signal processing

- This module contains several signal processing
functionalities to be applied to simulation/data
- operations are meant to be on the signal traces 
for individual antennas, suitable to be used both
in Grandlib format/ read from hdf5 files
- expects signal traces to be of the size (3,lengthoftrace)
"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, butter, lfilter
import scipy.fft as sf
from scipy import interpolate

logger = getLogger(__name__)


def get_filter(time, trace, fr_min, fr_max):
    """!
    Filter signal  in given bandwidth

    @note
      At present Butterworth filter only is implemented, others: what
      is close to hardware filter?

    :param time (array): [ns] time
    :param trace (array): ElectricField (muV/m)/voltage (muV) vectors to be filtered
    :param fr_min (float): [Hz] The minimal frequency of the bandpass filter
    :param fr_max (float): [Hz] The maximal frequency of the bandpass filter

    @returns: filtered trace in time domain
    """
    tstep = (time[1] - time[0]) * 1e-09  # s
    rate = 1 / tstep
    nyq = 0.5 * rate  # Nyquist limit
    low = fr_min / nyq
    high = fr_max / nyq
    order = 5
    coeff_b, coeff_a = butter(order, [low, high], btype="band")
    filtered = lfilter(coeff_b, coeff_a, trace)  # this is data in the time domain
    return filtered


def get_peakamptime_hilbert(time, trace, f_min, f_max, filtered=False):
    """!
    Get Peak and time of EField trace, either filtered or unfiltered

    :param time (array): time
    :param trace (array): ElectricField (muV/m) vectors to be filtered- expected
      size (3,dlength), dlength= size of the signal trace
    :param f_min (float): [Hz] The minimal frequency of the bandpass filter
    :param f_max (float): [Hz] The maximal frequency of the bandpass filter
    :param filtered (bool): if true filtering is applied else raw input trace is used

    @return TBD
    """
    if filtered:
        logger.debug("filtering the signal .....")
        filt = get_filter(time, trace, f_min, f_max)
    else:
        logger.debug("continuing with raw signal ...")
        filt = trace
    hilbert_amp = np.abs(hilbert(filt, axis=-1))
    peakamp = max([max(hilbert_amp[0]), max(hilbert_amp[1]), max(hilbert_amp[2])])
    peaktime = time[np.argwhere(hilbert_amp == peakamp)[0][1]]
    return peaktime, peakamp


def get_fastest_size_fft(sig_size, f_samp_mhz, padding_fact=1):
    """

    :param sig_size:
    :param f_samp_mhz:
    :param padding_fact:

    @return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_fact >= 1
    dt_s = 1e-6 / f_samp_mhz
    fastest_size_fft = sf.next_fast_len(int(padding_fact * sig_size + 0.5))
    freqs_mhz = sf.rfftfreq(fastest_size_fft, dt_s) * 1e-6
    return fastest_size_fft, freqs_mhz


def interpol_at_new_x(a_x, a_y, new_x):
    """!
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    @return F(new_x) (float, (M)): interpolation of F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)
