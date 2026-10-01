"""Peak amplitude and peak time extraction from antenna traces."""

# Created by Marion Guelfand at 19/01/2026

import numpy as np
from scipy.signal import hilbert
from grand.basis import validate as _validate

def _trace_channels(trace, channels, where):
    r"""Checks a (channels, samples) trace and the channel indices to use."""
    trace = _validate.as_array(trace, "trace", where, ndim=2)
    if trace.shape[1] == 0:
        raise ValueError(_validate.message(where, "'trace' has no samples"))
    # A slice or a boolean mask selects channels as NumPy does; only
    # explicit indices are checked here.
    if isinstance(channels, slice):
        return trace
    if np.asarray(channels).dtype == bool:
        if np.asarray(channels).shape != (trace.shape[0],):
            raise ValueError(_validate.message(
                where, "a boolean 'channels' mask needs one value per channel (%d), got shape %s"
                % (trace.shape[0], np.asarray(channels).shape)))
        return trace
    for ch in np.atleast_1d(channels):
        index = _validate.as_integer(ch, "channels", where)
        if not -trace.shape[0] <= index < trace.shape[0]:
            raise ValueError(_validate.message(
                where, "channel %d does not exist: 'trace' has %d channels (0 to %d)"
                % (index, trace.shape[0], trace.shape[0] - 1)))
    return trace


def get_peak_amplitude(trace, channels, return_envelope=False):
    """
    Compute the peak amplitude of a trace (in ADC counts or uV or uV/m).

    The envelope is the Euclidean norm of the Hilbert envelopes of the
    selected components, sqrt(sum_i |hilbert(ch_i)|^2).  (It was the Hilbert
    envelope of the norm, which is biased by -4 % to +6 % depending on the
    carrier frequency, #288.)

    Parameters
    ----------
    trace : np.ndarray
        2D array of shape (n_channels, n_samples).
    channels : list or array-like
        Indices of the channels to use for the amplitude computation.
    return_envelope : bool, optional
        If True, also return the Hilbert envelope of the signal.

    Returns
    -------
    peak_amp : float
        Maximum amplitude of the Hilbert envelope.
    hilbert_amp : np.ndarray, optional
        Hilbert envelope of the signal (only if return_envelope=True).
    """
    trace = _trace_channels(trace, channels, "get_peak_amplitude")
    selected = trace[channels, :]
    hilbert_amp = np.sqrt(np.sum(np.abs(hilbert(selected, axis=-1)) ** 2, axis=0))
    peak_amp = np.max(hilbert_amp)
    if return_envelope:
        return peak_amp, hilbert_amp
    return peak_amp

def compute_t0(t_object):
    """
    Compute the initial time (t0) for each trace in nanoseconds.

    Parameters
    ----------
    t_object : object
        Object containing 'du_seconds' and 'du_nanoseconds' attributes.

    Returns
    -------
    np.ndarray
        Array of t0 values for each trace in nanoseconds.
    """
    event_second = np.array(t_object.du_seconds).min()
    event_nano = np.array(t_object.du_nanoseconds).min()
    t0 = (np.array(t_object.du_seconds) - event_second) * 1e9 - event_nano + np.array(t_object.du_nanoseconds)
    return t0


def get_peak_time(trace, t0, channels, dt_ns=2):
    """
    Compute the time of the peak signal amplitude.

    The peak time is obtained from the maximum of the Hilbert envelope
    and converted from sample index to physical time.

    Parameters
    ----------
    trace : np.ndarray
        2D array of shape (n_channels, n_samples).
    t0 : float or np.ndarray
        Initial time offset(s) in nanoseconds.
    channels : list or array-like
        Indices of the channels to use.
    dt_ns : float, optional
        Sampling time in nanoseconds (default is 2 ns).

    Returns
    -------
    float or np.ndarray
        Time of the peak amplitude in seconds.
    """
    _validate.plausible(dt_ns, "dt_ns", "get_peak_time", "time_step_ns")   # seconds were taken as ns (#266)
    _validate.positive(_validate.as_real(dt_ns, "dt_ns", "get_peak_time"), "dt_ns", "get_peak_time", "ns")
    _, hilbert_amp = get_peak_amplitude(trace, channels, return_envelope=True)
    peak_idx = np.argmax(hilbert_amp)
    #Convert sample index to time in seconds (2ns per sample)
    peak_time = (peak_idx * dt_ns + t0) * 1e-9  
    return peak_time

def convert_voltage_to_ADC(trace, channels, adc_full_scale=8192, voltage_ref=0.9):
    """
    Convert voltage traces to ADC counts.

    Parameters
    ----------
    trace : np.ndarray
        2D array of voltage traces in microvolts (µV).
    channels : int, list, slice or boolean mask
        The channels to convert, selected as NumPy indexing does.
    adc_full_scale : int, optional
        ADC full scale value (default: 8192).
    voltage_ref : float, optional
        Reference voltage in volts (default: 0.9 V).

    Returns
    -------
    np.ndarray
        Trace converted to ADC counts.
    """
    trace = _trace_channels(trace, channels, "convert_voltage_to_ADC")
    _validate.positive(_validate.as_real(voltage_ref, "voltage_ref", "convert_voltage_to_ADC"),
                       "voltage_ref", "convert_voltage_to_ADC", "V")
    ADC_trace = trace.copy().astype(float) 
    
    # Index as NumPy does, so a mask, a slice or a single index selects the
    # same channels as in get_peak_amplitude (issue #263)
    ADC_trace[channels, :] = trace[channels, :] * 1e-6 * adc_full_scale / voltage_ref
            
    return ADC_trace


