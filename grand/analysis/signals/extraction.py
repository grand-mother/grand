# Created by Marion Guelfand at 19/01/2026

import numpy as np
from scipy.signal import hilbert, convolve

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

def compute_t0_sims(adc_tree, jitter=0.0, seed=71):
    """
    Compute the initial time of each simulated ADC trace.

    Parameters
    ----------
    adc_tree : TADC
        Simulation ADC tree (du_seconds, du_nanoseconds, run_number, event_number).
    jitter : float, optional
        Standard deviation of an additional Gaussian timing jitter, in ns
        (default 0: no jitter). One independent draw per DU, rounded to the
        2 ns sampling step. The ADC traces are not modified.
    seed : int, optional
        Extra entropy of the jitter generator, e.g. zlib.crc32(file name), so
        that equal (run, event) numbers in different files get different draws.
        The generator is np.random.default_rng([seed, run_number, event_number]):
        the jitter is reproducible and the same N(0, 1) draws are used for
        every jitter value.

    Returns
    -------
    t0_ns : np.ndarray
        Initial times in ns, relative to the earliest DU of the event.
    """
    if jitter < 0:
        raise ValueError(f"jitter must be >= 0 ns, got {jitter}")

    seconds = np.asarray(adc_tree.du_seconds, dtype=np.int64)
    nanoseconds = np.asarray(adc_tree.du_nanoseconds, dtype=np.float64)
    if seconds.size == 0:
        return np.zeros(0)

    # Relative time (ns): integer seconds first, then float64 on small values only
    t_ns = (seconds - seconds.min()).astype(np.float64) * 1e9 + nanoseconds
    t0_ns = t_ns - t_ns.min()

    if jitter > 0:
        rng = np.random.default_rng(
            [int(seed), int(adc_tree.run_number), int(adc_tree.event_number)]
        )
        z = rng.standard_normal(t0_ns.size)
        # Gaussian timing jitter (ns), rounded to the 2 ns sampling step
        t0_ns = t0_ns + np.round(jitter * z / 2) * 2

    return t0_ns

def get_peak_time_efield(trace, t0, channels, dt_ns=2):
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
    # Convert sample index to time in seconds (2ns per sample by default)
    peak_time = (peak_idx * dt_ns + t0) * 1e-9  
    return peak_time

def get_peak_time_adc(trace, nutrig_template, t0, dt_ns=2):
    """
    Compute the time of the peak signal amplitude for ADC traces.

    Parameters
    ----------
    trace : np.ndarray
        1D array of shape (n_samples,).
    nutrig_template : np.ndarray
        Template signal from the NUTRIG analysis.
    t0 : float or np.ndarray
        Initial time offset(s) in nanoseconds for the antenna.
    dt_ns : float, optional
        Sampling time in nanoseconds (default is 2 ns).

    Returns
    -------
    float or np.ndarray
        Time of the peak amplitude in seconds.
    """
    trace = trace[np.argmax(np.max(trace, axis=1))] 
    convolution = convolve(trace, nutrig_template, mode='same')
    convolution_hilbert = np.abs(hilbert(convolution))
    peak_idx = np.argmax(convolution_hilbert)
    # Convert sample index to time in seconds (2ns per sample by default)
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


