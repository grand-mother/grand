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
    r"""Returns the interpolated maximum of a trace, from a parabola through 3 points.

    Fits :math:`ax^2 + bx + c` through the largest sample and its two
    neighbours, offset to :math:`(x_0, y_0)`, and returns the vertex:

    .. math::

        x_m = x_0 - \frac{b}{2a}, \qquad y_m = y_0 - \frac{b^2}{4a}

    The trace is assumed non-negative, and ``idx_max`` is assumed to be a
    genuine local maximum with a sample on either side.

    Parameters
    ----------
    x_trace : ndarray
        Sample positions.
    y_trace : ndarray
        Sample values.
    idx_max : int
        Index of the largest sample.

    Returns
    -------
    tuple of float
        Position and value of the interpolated maximum, from a parabola through three points.

    Examples
    --------
    A peak between two samples is recovered more precisely than the sample grid
    allows.

    .. jupyter-execute::

        import numpy as np
        from grand.basis.signal import find_max_with_parabola_interp_3pt

        t = np.arange(512) * 0.5
        true_peak = 100.25                              # deliberately between samples
        y = np.exp(-((t - true_peak) ** 2) / (2 * 6.0 ** 2))

        idx = int(np.argmax(y))
        t_interp, v = find_max_with_parabola_interp_3pt(t, y, idx)
        print("nearest sample : %.4f ns" % t[idx])
        print("interpolated   : %.4f ns" % t_interp)
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


def find_max_with_parabola_interp(x_trace, y_trace, idx_max, factor_hill=0.96):
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

    Parameters
    ----------
    x_trace : ndarray
        Sample positions.
    y_trace : ndarray
        Sample values.
    idx_max : int
        Index of the largest sample.
    factor_hill : float, optional
        Fraction of the peak defining how much of the hill to fit.

    Returns
    -------
    tuple of float
        Position and value of the interpolated maximum.
    """
    # y threshold mean around max (so 3 points) * factor_hill
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
    while e_idx < (nb_sple-1) and nb_out < out_lim:
        if y_trace[e_idx] < y_lim:
            nb_out += 1
        else:
            nb_out = 0
            last_idx = e_idx
        e_idx += 1
    e_idx = last_idx
    if e_idx >= nb_sple:
        e_idx = nb_sple -1
    logger.debug(f"border around idx max {idx_max} is {b_idx}, {e_idx}")
    logger.debug(f"{x_trace[b_idx]}\t{x_trace[e_idx]}")
    if (e_idx - b_idx) <= 2:
        return find_max_with_parabola_interp_3pt(x_trace, y_trace, idx_max)
    logger.debug("Parabola interp: mode hill")
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

    Parameters
    ----------
    time : ndarray
        Time axis, in nanoseconds.
    trace : ndarray
        Trace to filter.
    fr_min : float
        Lower band edge, in **Hz**.  Note the unit: most frequency arguments
        in this package are in MHz, and this one is not -- passing 50 rather
        than 50e6 silently returns an all-zero trace.
    fr_max : float
        Upper band edge, in **Hz**.

    Returns
    -------
    ndarray
        The band-passed trace.

    Examples
    --------
    The band edges are in **hertz**.  Passing megahertz does not raise; it
    silently returns an empty trace, which is the failure most easily mistaken
    for a quiet event.

    .. jupyter-execute::

        import numpy as np
        from grand.basis.signal import get_filter

        t = np.arange(1024) * 0.5                       # ns
        tone = np.sin(2 * np.pi * 100.0 * t * 1e-3)     # a 100 MHz tone

        kept = get_filter(t, tone, 50e6, 200e6)         # edges in Hz
        print("in band, kept: %.3f" % np.std(kept))

        # The same numbers read as MHz pass nothing at all -- no error, just zeros.
        print("edges given in MHz: %.3f" % np.std(get_filter(t, tone, 50.0, 200.0)))
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

    Parameters
    ----------
    a2_time : ndarray
        Time axis per trace.
    a3_trace : ndarray, shape (n_du, 3, n_samples)
        Traces.

    Returns
    -------
    tuple of ndarray
        Peak time, peak amplitude, the norm, and its Hilbert envelope.
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

    Parameters
    ----------
    sig_size : int
        Length of the trace, in samples.
    f_samp_mhz : float or ndarray
        Sampling frequency, in MHz.
    padding_fact : float, optional
        Zero-padding factor; at least 1.

    Returns
    -------
    tuple
        Transform length, and the frequency axis in MHz.
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

    Parameters
    ----------
    a_x : ndarray
        Sample positions.
    a_y : ndarray
        Sample values.
    new_x : ndarray
        Positions to interpolate onto.

    Returns
    -------
    ndarray
        Interpolated values, zero outside the range of `a_x`.

    Examples
    --------
    Values outside the measured range come back as zero rather than
    extrapolated: the tabulated antenna and RF-chain data have no meaning
    beyond 30-250 MHz.

    .. jupyter-execute::

        import numpy as np
        from grand.basis.signal import interpol_at_new_x

        a_x = np.linspace(30.0, 250.0, 100)             # the measured band, in MHz
        a_y = np.ones_like(a_x)

        print(interpol_at_new_x(a_x, a_y, np.array([10.0, 140.0, 400.0])))
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)
