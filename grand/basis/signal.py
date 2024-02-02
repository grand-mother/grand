"""
* This module contains several signal processing functionalities to be applied to simulation/data
* operations are meant to be on the signal traces for individual antennas, suitable to be used both
  in Grandlib format/ read from hdf5 files 
* expects signal traces to be of the size (3,lengthoftrace)
"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, butter, lfilter
import scipy.fft as sf
from scipy import interpolate

logger = getLogger(__name__)


def find_max_with_parabola_interp_3pt(x_trace, y_trace, idx_max):
    """Parabolic interpolation of the maximum with 3 points


    trace : all values >= 0

    :param x_trace:
    :param y_trace:
    algo Mode pic, input 3 values and the middle one is max:
        parabola : ax^2 + bx + c
        offset of (x0, y0)
        solve coef a, b , interpolation of the maximum is
          x_m = x0 - b/2a
          y_m = y0 - b^2/4a
    :param idx_max: index of sample max, idx_max < nb_sample
    :type idx_max: int
    :return: x_max, y_max
    """
    if (idx_max >= len(x_trace) - 1) or idx_max == 0:
        return x_trace[idx_max], y_trace[idx_max]
    logger.debug(f"Parabola interp: mode pic {idx_max} {len(x_trace)}")
    # remove offset (x0, v0)
    y_pic = y_trace[idx_max : idx_max + 2] - y_trace[idx_max - 1]
    x_pic = x_trace[idx_max : idx_max + 2] - x_trace[idx_max - 1]
    logger.debug(x_trace[idx_max : idx_max + 2])
    logger.debug(y_trace[idx_max : idx_max + 2])
    # solve coef a, b
    r_pic = y_pic / x_pic
    c_a = (r_pic[1] - r_pic[0]) / (x_pic[1] - x_pic[0])
    c_b = r_pic[0] - c_a * x_pic[0]
    # interpolation of the maximum is
    x_m = -c_b / (2 * c_a)
    x_max = x_trace[idx_max - 1] + x_m
    y_max = y_trace[idx_max - 1] + x_m * c_b / 2
    return x_max, y_max


def find_max_with_parabola_interp(x_trace, y_trace, idx_max, factor_hill=0.8):
    """Parabolic interpolation of the maximum with more than 3 points

    trace : all values >= 0

    algo:
      1. find begin idx, ie trace[--idx_max] > v_max*factor_hill
      2. find end idx, ie trace[idx_max++] > v_max*factor_hill
      3. if nb idx <= 2 : mode pic else mode hill
      4. Mode pic : 3 values and the middle one is max
         4.1 offset of (x0, v0)
         4.2 solve coef a, b => x_m = offset - b/2a ; v_m=offset - b^2/4a
      5. Mode hill:
         5.0 offset of (x, y) of first sample
         5.1 solve overdetermined linear system with a, b, c
         5.2 x_m =offset - b/2a ; v_m=offset - b^2/4a + c

    :param trace:
    :type trace:
    :param idx_max:
    :type idx_max:
    :param factor_hill:
    :type factor_hill:
    """
    y_lim = (y_trace[idx_max - 1 : idx_max + 2].sum() / 3) * factor_hill
    logger.debug(f"y_lim={y_lim}")
    # 1
    b_idx = idx_max - 1
    out_lim = 6
    nb_out = 0
    last_idx = b_idx
    while b_idx > 0 and nb_out < out_lim:
        if y_trace[b_idx] < y_lim:
            nb_out += 1
        else:
            nb_out = 0
            last_idx = b_idx
        b_idx -= 1
    b_idx = last_idx
    # 2
    nb_sple = y_trace.shape[0]
    e_idx = idx_max + 1
    nb_out = 0
    last_idx = e_idx
    while e_idx < nb_sple and nb_out < out_lim:
        if y_trace[e_idx] < y_lim:
            nb_out += 1
        else:
            nb_out = 0
            last_idx = e_idx
        e_idx += 1
    e_idx = last_idx
    logger.debug(f"border around idx max {idx_max} is {b_idx}, {e_idx}")
    logger.debug(f"{x_trace[b_idx]}\t{x_trace[e_idx]}")
    if (e_idx - b_idx) <= 2:
        return find_max_with_parabola_interp_3pt(x_trace, y_trace, idx_max)
    else:
        logger.debug(f"Parabola interp: mode hill")
        # mode hill
        y_hill = y_trace[b_idx : e_idx + 1] - y_trace[b_idx]
        x_hill = x_trace[b_idx : e_idx + 1] - x_trace[b_idx]
        mat = np.empty((x_hill.shape[0], 3), dtype=np.float32)
        mat[:, 2] = 1
        mat[:, 1] = x_hill
        mat[:, 0] = x_hill * x_hill
        sol = np.linalg.lstsq(mat, y_hill, rcond=None)[0]
        if -1e-5 < sol[0] and sol[0] < 1e-5:
            # very flat case
            return x_trace[idx_max], y_trace[idx_max]
        x_m = -sol[1] / (2 * sol[0])
        x_max = x_trace[b_idx] + x_m
        y_max = y_trace[b_idx] + x_m * sol[1] / 2 + sol[2]
        return x_max, y_max


def get_filter(time, trace, fr_min, fr_max):
    """
    Filter signal  in given bandwidth

    @note
      At present Butterworth filter only is implemented, others: what
      is close to hardware filter?

    :param time (array): [ns] time
    :param trace (array): ElectricField (muV/m)/voltage (muV) vectors to be filtered
    :param fr_min (float): [Hz] The minimal frequency of the bandpass filter
    :param fr_max (float): [Hz] The maximal frequency of the bandpass filter

    :return: filtered trace in time domain
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


def get_peakamptime_norm_hilbert(a2_time, a3_trace):
    """
    Get peak Hilbert amplitude norm of trace (v_max) and its time t_max without interpolation

    :param time (D,S): time, with D number of vector of trace, S number of sample
    :param traces (D,3,S): trace

    :return: t_max float(D,) v_max float(D,), norm_hilbert_amp float(D,S),
            idx_max int, norm_hilbert_amp float(D,S)
    """
    hilbert_amp = np.abs(hilbert(a3_trace, axis=-1))
    norm_hilbert_amp = np.linalg.norm(hilbert_amp, axis=1)
    # add dimension for np.take_along_axis()
    idx_max = np.argmax(norm_hilbert_amp, axis=1)[:, np.newaxis]
    t_max = np.take_along_axis(a2_time, idx_max, axis=1)
    v_max = np.take_along_axis(norm_hilbert_amp, idx_max, axis=1)
    # remove dimension (np.squeeze) to have ~vector ie shape is (n,) instead (n,1)
    return np.squeeze(t_max), np.squeeze(v_max), idx_max, norm_hilbert_amp


def get_fastest_size_fft(sig_size, f_samp_mhz, padding_fact=1):
    """
    #RK: This function is copied to grand/simu/master_simu.py where it is used. Remove it from here if it is not used anywhere else.

    :param sig_size:
    :param f_samp_mhz:
    :param padding_fact:

    :return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_fact >= 1
    dt_s = 1e-6 / f_samp_mhz
    fastest_size_fft = sf.next_fast_len(int(padding_fact * sig_size + 0.5))
    freqs_mhz = sf.rfftfreq(fastest_size_fft, dt_s) * 1e-6
    return fastest_size_fft, freqs_mhz


def interpol_at_new_x(a_x, a_y, new_x):
    """
    #RK: This function is copied to galaxy.py and rf_chain.py where it is used. Remove it from here if it is not used anywhere else.

    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)
