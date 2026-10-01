# -*- coding: utf-8 -*-
r"""Geometry helpers on arrays of directions."""

import numpy as np
import pytest


def test_sin_geomag_angle_is_per_direction():
    r"""#214: an array of angles gave one number, above 1."""
    from grand.analysis.geom.angles import sin_geomag_angle

    theta, phi = np.array([0.5, 1.0, 1.4]), np.array([1.0, 2.0, -0.3])
    together = sin_geomag_angle(theta, phi)
    assert together.shape == (3,)
    assert together == pytest.approx([sin_geomag_angle(t, p) for t, p in zip(theta, phi)])
    assert isinstance(sin_geomag_angle(0.5, 1.0), float)


@pytest.mark.parametrize("carrier_mhz", [50.0, 80.0, 130.0, 190.0])
def test_peak_amplitude_is_the_envelope_of_the_field(carrier_mhz):
    r"""#288: the Hilbert envelope of the norm was biased by -4 % to +6 %."""
    from grand.analysis.signals.extraction import get_peak_amplitude

    t = np.arange(2048) * 0.5e-3            # us, 2 GHz
    envelope = 100.0 * np.exp(-0.5 * ((t - 0.5) / 0.05) ** 2)
    carrier = envelope * np.sin(2 * np.pi * carrier_mhz * t)
    trace = np.array([0.6 * carrier, 0.8 * carrier, 0 * carrier])
    assert get_peak_amplitude(trace, [0, 1, 2]) == pytest.approx(100.0, rel=2e-3)
