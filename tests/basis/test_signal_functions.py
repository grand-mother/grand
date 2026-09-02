# -*- coding: utf-8 -*-
r"""Numerical checks on the signal helpers.

``grand.basis.signal`` holds the peak-finding, filtering and interpolation
that the trace handling and the voltage simulator both depend on, and it was
40% covered.  Each test below states a property the function must have, so a
failure names the defect rather than merely reporting a changed number.
"""

import numpy as np
import pytest

import grand.basis.signal as gds

F_SAMP_MHZ = 2000.0


def _gaussian(n=512, centre=200.0, width=6.0, dt=0.5):
    r"""Returns a time axis and a Gaussian pulse on it.

    Parameters
    ----------
    n : int, optional
        Number of samples.
    centre : float, optional
        Peak position, in nanoseconds.
    width : float, optional
        Standard deviation, in nanoseconds.
    dt : float, optional
        Sample interval, in nanoseconds.

    Returns
    -------
    tuple of ndarray
        Time axis and pulse.
    """
    t = np.arange(n) * dt
    return t, np.exp(-((t - centre) ** 2) / (2 * width ** 2))


def test_parabola_interpolation_beats_the_sample_grid():
    r"""The interpolated peak is closer to the true one than the best sample.

    A Gaussian centred between two samples has its true maximum off the grid.
    Three-point parabolic interpolation exists precisely to recover that, so
    it must do better than simply taking the largest sample.
    """
    dt, centre = 0.5, 200.25          # deliberately between samples
    t, y = _gaussian(centre=centre, dt=dt)
    idx = int(np.argmax(y))

    t_grid = t[idx]
    t_interp, _ = gds.find_max_with_parabola_interp_3pt(t, y, idx)

    assert abs(t_interp - centre) < abs(t_grid - centre), (
        'interpolation (%.4f) is no better than the grid (%.4f) for a true '
        'peak at %.4f' % (t_interp, t_grid, centre))
    assert abs(t_interp - centre) < dt / 2


def test_parabola_interpolation_recovers_the_amplitude():
    r"""The interpolated peak value is at least the largest sample."""
    t, y = _gaussian(centre=200.25)
    idx = int(np.argmax(y))
    _, v = gds.find_max_with_parabola_interp_3pt(t, y, idx)
    assert v >= y[idx] - 1e-12, 'interpolated peak is below the sampled peak'
    assert v <= 1.01, 'interpolated peak exceeds the true amplitude'


def test_bandpass_removes_out_of_band_power():
    r"""A tone outside the pass band is suppressed; one inside survives.

    Both are checked, because a filter that removed everything would pass the
    first assertion alone.
    """
    n, dt = 1024, 0.5
    t = np.arange(n) * dt
    in_band = np.sin(2 * np.pi * 100.0 * t * 1e-3)    # 100 MHz
    out_band = np.sin(2 * np.pi * 400.0 * t * 1e-3)   # 400 MHz

    # The band edges are in Hz, not MHz -- see the note in the docstring.
    kept = gds.get_filter(t, in_band, 50e6, 200e6)
    removed = gds.get_filter(t, out_band, 50e6, 200e6)

    assert np.std(kept) > 0.5 * np.std(in_band), 'the pass band was attenuated'
    assert np.std(removed) < 0.2 * np.std(out_band), (
        'a 400 MHz tone survived a 50-200 MHz pass band')


def test_bandpass_edges_are_in_hertz():
    r"""Band edges given in MHz silently produce an empty trace.

    ``fr_min`` and ``fr_max`` are in Hz while nearly every other frequency
    argument in the package is in MHz.  Passing 50 instead of 50e6 does not
    raise -- it returns zeros, which is the failure mode most likely to be
    mistaken for a quiet event.  This pins the behaviour so that a future
    change of unit is a deliberate one.
    """
    n, dt = 1024, 0.5
    t = np.arange(n) * dt
    tone = np.sin(2 * np.pi * 100.0 * t * 1e-3)

    assert np.std(gds.get_filter(t, tone, 50e6, 200e6)) > 0.5
    assert np.std(gds.get_filter(t, tone, 50.0, 200.0)) == pytest.approx(0.0, abs=1e-9)


def test_interpolation_is_zero_outside_the_input_range():
    r"""Values beyond the sampled range are zero, not extrapolated.

    The tabulated antenna and RF-chain data are measured over 30-250 MHz and
    have no meaning outside it, so extrapolating would invent a response.
    """
    a_x = np.linspace(30.0, 250.0, 100)
    a_y = np.ones_like(a_x)
    new_x = np.array([10.0, 30.0, 140.0, 250.0, 400.0])

    out = gds.interpol_at_new_x(a_x, a_y, new_x)
    assert out[0] == 0.0, 'extrapolated below the range'
    assert out[-1] == 0.0, 'extrapolated above the range'
    assert np.isclose(out[2], 1.0), 'interpolation inside the range is wrong'


def test_fft_size_is_at_least_the_requested_length():
    r"""The transform length is never shorter than what was asked for."""
    for size in (100, 512, 1000, 1001):
        n, freqs = gds.get_fastest_size_fft(size, F_SAMP_MHZ)
        assert n >= size, 'padded to %d, shorter than the input %d' % (n, size)
        assert freqs.size == n // 2 + 1, 'frequency axis does not match rfft'


def test_padding_improves_frequency_resolution():
    r"""Padding halves the bin spacing when the factor is two."""
    _, coarse = gds.get_fastest_size_fft(1000, F_SAMP_MHZ, padding_fact=1)
    _, fine = gds.get_fastest_size_fft(1000, F_SAMP_MHZ, padding_fact=2)
    assert fine[1] < coarse[1], 'padding did not improve the resolution'
    assert np.isclose(fine[1], coarse[1] / 2, rtol=0.05)


def test_fft_frequency_axis_reaches_nyquist():
    r"""The axis runs from zero to the Nyquist frequency."""
    _, freqs = gds.get_fastest_size_fft(512, F_SAMP_MHZ)
    assert np.isclose(freqs[0], 0.0)
    assert np.isclose(freqs[-1], F_SAMP_MHZ / 2, rtol=1e-6)


def test_hilbert_peak_finds_the_pulse():
    r"""The Hilbert envelope peaks where the pulse is, on all three arms."""
    n_du, dt = 2, 0.5
    t, pulse = _gaussian(centre=150.0, dt=dt)
    traces = np.stack([np.stack([pulse, 0.5 * pulse, 0.2 * pulse])
                       for _ in range(n_du)])
    times = np.stack([t for _ in range(n_du)])

    t_peak, amplitude = gds.get_peakamptime_norm_hilbert(times, traces)[:2]
    assert np.allclose(t_peak, 150.0, atol=2.0), (
        'peak found at %s, expected 150 ns' % t_peak)
    assert (amplitude > 0).all()
