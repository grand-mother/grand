# -*- coding: utf-8 -*-
r"""`Efield2Voltage.final_resample` keeps amplitude and frequency (#270).

The branch was never executed by a test, so dropping the amplitude
renormalisation or truncating one sample short passed.
"""

import numpy as np
import pytest
import scipy.fft as sf

from grand.sim.efield2voltage import Efield2Voltage

RATE, N, F_MHZ, AMPLITUDE, DURATION_US = 2000.0, 1024, 50.0, 3.0, 0.4


def _resampled(target_rate):
    e2v = Efield2Voltage.__new__(Efield2Voltage)
    t = np.arange(N) / RATE                      # us
    trace = AMPLITUDE * np.sin(2 * np.pi * F_MHZ * t)
    e2v.vout = np.tile(trace, (1, 3, 1))
    e2v.vout_f = sf.rfft(e2v.vout)
    e2v.nb_du = 1
    e2v.fft_size = N
    e2v.f_samp_mhz = np.array([RATE])
    e2v.target_sampling_rate_mhz = target_rate
    e2v.target_duration_us = DURATION_US
    e2v.target_lenght = N
    e2v.params = dict(add_noise=False, add_rf_chain=False, add_rf_chain_nut=False,
                      add_rf_chain_gaa=False)
    e2v.final_resample()
    return e2v.vout[0, 0]


@pytest.mark.parametrize("target_rate", [4000.0, 1000.0])
def test_resampling_keeps_amplitude_frequency_and_length(target_rate):
    out = _resampled(target_rate)
    assert out.size == int(DURATION_US * target_rate)
    assert np.max(np.abs(out)) == pytest.approx(AMPLITUDE, rel=0.02)
    freqs = sf.rfftfreq(out.size, 1 / target_rate)
    assert freqs[np.argmax(np.abs(sf.rfft(out)))] == pytest.approx(F_MHZ, abs=freqs[1])
