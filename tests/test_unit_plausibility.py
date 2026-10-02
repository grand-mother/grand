# -*- coding: utf-8 -*-
r"""Values almost certainly in the wrong unit warn (#266)."""

import numpy as np
import pytest

from grand.basis.validate import GRANDlibWarning


def test_frequencies_in_hz_warn():
    from grand.sim.noise.galaxy import galactic_noise

    freqs_mhz = np.fft.rfftfreq(1024, 0.5e-3)
    with pytest.warns(GRANDlibWarning, match="freqs_mhz.*Hz or GHz"):
        galactic_noise(18.0, 1024, freqs_mhz * 1e6, 2, seed=1)


def test_sampling_rate_and_time_step_warn():
    from grand.analysis.signals.extraction import get_peak_time
    from grand.basis.signal import get_fastest_size_fft
    from grand.sim.detector.adc import ADC

    with pytest.warns(GRANDlibWarning, match="input_sampling_rate_mhz"):
        ADC().downsample(np.ones((1, 3, 64)), input_sampling_rate_mhz=500e6)
    with pytest.warns(GRANDlibWarning, match="f_samp_mhz"):
        get_fastest_size_fft(100, 2e9)
    with pytest.warns(GRANDlibWarning, match="dt_ns.*seconds"):
        try:
            get_peak_time(np.ones((1, 3, 64)), 1e-7, [0, 1], dt_ns=2e-9)
        except Exception:
            pass


def test_degrees_for_radians_and_ns_for_seconds_warn():
    from grand.analysis.fitting.plane_wave import PWF_semianalytical
    from grand.analysis.geom.angles import sin_geomag_angle

    with pytest.warns(GRANDlibWarning, match="angle in degrees"):
        sin_geomag_angle(45.0, 60.0)
    xants = np.array([[0.0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 10]])
    tants = np.array([0.0, 1e-6, 2e-6, 3e-6])
    with pytest.warns(GRANDlibWarning, match="in ns rather than s"):
        PWF_semianalytical(xants, tants * 1e9)


def test_plausible_values_do_not_warn(recwarn):
    from grand.analysis.geom.angles import sin_geomag_angle
    from grand.basis.signal import get_fastest_size_fft

    sin_geomag_angle(0.5, 1.0)
    get_fastest_size_fft(100, 2000.0)
    assert not [w for w in recwarn if issubclass(w.category, GRANDlibWarning)]
