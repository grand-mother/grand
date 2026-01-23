# Created by Marion Guelfand at 19/01/2026

import numpy as np
from scipy.signal import hilbert

def get_peak_amplitude(trace, channels, return_envelope=False):
    """
    Compute the peak amplitude of a trace (in ADC counts or uV or uV/m).

    The signal amplitude is computed as the Euclidean norm of the selected
    components (x, y, z), followed by a Hilbert transform to obtain
    the signal envelope.

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
    trace = np.asarray(trace)
    selected = trace[channels, :]
    E_modulus = np.linalg.norm(selected, axis=0)
    hilbert_amp = np.abs(hilbert(E_modulus))
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
    channels : list or array-like
        Indices of the channels to convert.
    adc_full_scale : int, optional
        ADC full scale value (default: 8192).
    voltage_ref : float, optional
        Reference voltage in volts (default: 0.9 V).

    Returns
    -------
    np.ndarray
        Trace converted to ADC counts.
    """
    trace = np.asarray(trace)  
    ADC_trace = trace.copy().astype(float) 
    
    for ch in channels:
        ADC_trace[ch, :] = trace[ch, :] * 1e-6 * adc_full_scale / voltage_ref
            
    return ADC_trace


