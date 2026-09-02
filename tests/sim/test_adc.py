# -*- coding: utf-8 -*-
r"""Checks on the ADC model: quantisation, saturation and downsampling.

``grand.sim.detector.adc`` is the last stage of the chain and was 28%
covered.  It is where a continuous voltage becomes the integer counts a
``TADC`` tree holds, so its failure modes -- silent clipping, a wrong bit
depth, a resampling that loses the pulse -- change recorded data rather than
raising.
"""

import numpy as np
import pytest

from grand.sim.detector.adc import ADC


@pytest.fixture(scope='module')
def adc():
    r"""Returns a default ADC model."""
    return ADC()


def _pulse(n=1024, dt_ns=0.5, centre_ns=200.0, width_ns=5.0, amplitude=1.0):
    r"""Returns a Gaussian voltage pulse of the given amplitude, in microvolts.

    Parameters
    ----------
    n : int, optional
        Number of samples.
    dt_ns : float, optional
        Sample interval, in nanoseconds.
    centre_ns, width_ns : float, optional
        Peak position and width, in nanoseconds.
    amplitude : float, optional
        Peak value, in microvolts.

    Returns
    -------
    ndarray, shape (1, 3, n)
        One unit, three arms.
    """
    t = np.arange(n) * dt_ns
    p = amplitude * np.exp(-((t - centre_ns) ** 2) / (2 * width_ns ** 2))
    return np.stack([np.stack([p, 0.5 * p, 0.2 * p])])


def test_digitised_output_is_integral(adc):
    r"""ADC counts are whole numbers.

    A float output would mean the quantisation step had been skipped, which
    is easy to miss because the values still look reasonable.
    """
    counts = adc.process(_pulse(amplitude=100.0))
    counts = np.asarray(counts)
    assert np.allclose(counts, np.round(counts)), 'output is not quantised'


def test_saturation_clips_rather_than_wrapping(adc):
    r"""A signal far above full scale is clipped, not wrapped around.

    Wrapping would turn a very large pulse into a small one of the opposite
    sign -- a failure that looks like data rather than like an error.
    """
    modest = np.asarray(adc.process(_pulse(amplitude=1e3)))
    enormous = np.asarray(adc.process(_pulse(amplitude=1e9)))

    assert np.abs(enormous).max() >= np.abs(modest).max(), (
        'a larger input gave a smaller output: the range wrapped')
    assert np.isfinite(enormous).all()


def test_saturation_is_bounded(adc):
    r"""No output exceeds the converter's range, however large the input."""
    counts = np.asarray(adc.process(_pulse(amplitude=1e12)))
    span = np.abs(counts).max()
    assert span < 2 ** 24, (
        'output reached %g, beyond any plausible converter range' % span)


def test_zero_input_gives_zero_counts(adc):
    r"""An identically zero trace digitises to zero."""
    counts = np.asarray(adc.process(np.zeros((1, 3, 512))))
    assert np.abs(counts).max() == 0.0


def test_downsampling_reduces_the_length(adc):
    r"""Resampling to a lower rate shortens the trace proportionally."""
    trace = _pulse(n=1024, dt_ns=0.5)          # 2000 MHz
    out = np.asarray(adc.downsample(trace, 2000.0))
    assert out.shape[-1] < trace.shape[-1], 'downsampling did not shorten the trace'
    assert out.shape[:-1] == trace.shape[:-1], 'downsampling changed the shape'


def test_downsampling_keeps_the_pulse(adc):
    r"""The pulse survives resampling, in position and in rough amplitude.

    A resampling that silently dropped or displaced the signal would leave a
    trace of the right length holding the wrong thing.
    """
    trace = _pulse(n=1024, dt_ns=0.5, centre_ns=200.0, width_ns=20.0)
    out = np.asarray(adc.downsample(trace, 2000.0))

    original_peak = np.argmax(trace[0, 0]) / trace.shape[-1]
    new_peak = np.argmax(out[0, 0]) / out.shape[-1]
    assert abs(original_peak - new_peak) < 0.05, (
        'the peak moved from %.3f to %.3f of the trace' % (original_peak, new_peak))
    assert np.abs(out).max() > 0.5 * np.abs(trace).max(), (
        'the pulse lost more than half its amplitude')


def test_larger_input_gives_larger_counts(adc):
    r"""Digitisation is monotonic below saturation.

    The amplitudes are well above the quantisation step; see
    :func:`test_quantisation_step_is_about_one_hundred_microvolts` for why
    10 and 100 microvolts would both give zero.
    """
    small = np.abs(np.asarray(adc.process(_pulse(amplitude=1e3)))).max()
    large = np.abs(np.asarray(adc.process(_pulse(amplitude=1e4)))).max()
    assert large > small, "a ten-fold larger input did not give more counts"


def test_quantisation_step_is_about_one_hundred_microvolts(adc):
    r"""One count is roughly 110 microvolts; anything smaller becomes zero.

    The converter maps 900 mV onto 8192 counts, so the least significant bit
    is about 110 microvolts. A signal below that is not small in the output,
    it is absent. Worth pinning: a simulation whose voltages come out under
    the step produces an all-zero trace, which looks like a quiet event
    rather than like a scaling error.
    """
    lsb = adc.max_voltage / adc.max_bit_value
    assert 50.0 < lsb < 200.0, "unexpected quantisation step: %.1f uV" % lsb

    below = np.abs(np.asarray(adc.process(_pulse(amplitude=lsb / 10)))).max()
    above = np.abs(np.asarray(adc.process(_pulse(amplitude=lsb * 100)))).max()
    assert below == 0.0, "a sub-LSB signal produced counts"
    assert above > 0.0, "a signal 100x the LSB produced none"
