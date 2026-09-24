# -*- coding: utf-8 -*-
r"""The reconstruction package recovers what it is given.

``grand/analysis/`` (Marion Guelfand, 2026) arrived without tests.  These are
round trips: arrival times are generated with the package's own forward
models for a known shower, fitted, and the fit must return the shower.  That
checks the fits are consistent with their models and with each other, and it
pins GRAND's "comes from" angle convention in the reconstruction, which
``tests/geo/test_angle_convention.py`` pins in the core transforms.

It does not check the models against nature -- that needs simulated or
measured events with a known answer, which is future work.

Two defects are pinned as behaviour rather than fixed, because the code is
the author's physics and changing it is her call:

- ``recons_swf`` accepts ``sigma`` and ignores it in both branches.  It does
  not change the fit, since a uniform scale does not move a minimum.
- ``recons_energy_from_voltage`` documents "float or array-like" but calls
  the builtin ``max``, so an array raises.
"""

import numpy as np
import pytest

pytest.importorskip('iminuit', reason='grand.analysis needs iminuit')

#: Directions (zenith, azimuth) in degrees across the range GRAND observes.
DIRECTIONS = [(35.0, 20.0), (62.0, 137.0), (80.0, 250.0), (85.0, 330.0)]


@pytest.fixture(scope='module')
def antennas():
    r"""Twelve antennas scattered over 6 km, near the reference altitude.

    Returns
    -------
    numpy.ndarray
        Positions in metres, shape (12, 3), in the frame the package uses:
        heights around ``constants.groundAltitude``.
    """
    import grand.analysis.constants as cons

    rng = np.random.default_rng(0)
    n = 12
    return np.column_stack([rng.uniform(-3000, 3000, n),
                            rng.uniform(-3000, 3000, n),
                            cons.groundAltitude + rng.uniform(-20, 20, n)])


@pytest.mark.parametrize('zenith, azimuth', DIRECTIONS)
def test_the_plane_wave_fit_recovers_the_direction(antennas, zenith, azimuth):
    r"""Times from ``PWF_model`` fit back to the direction that made them."""
    from grand.analysis.fitting.plane_wave import PWF_model, PWF_semianalytical

    truth = np.deg2rad([zenith, azimuth])
    times = PWF_model(truth, antennas)
    theta, phi = PWF_semianalytical(antennas, times)

    assert np.degrees(theta) == pytest.approx(zenith, abs=1e-6)
    assert np.degrees(phi) == pytest.approx(azimuth, abs=1e-6)


def test_the_plane_wave_loss_is_zero_at_the_truth(antennas):
    r"""The chi-square the fit minimises vanishes at the true direction."""
    from grand.analysis.fitting.plane_wave import PWF_loss, PWF_model

    truth = np.deg2rad([62.0, 137.0])
    times = PWF_model(truth, antennas)
    assert PWF_loss(truth, antennas, times, sigma=1e-9) == pytest.approx(0.0, abs=1e-9)
    off = truth + np.deg2rad([1.0, 0.0])
    assert PWF_loss(off, antennas, times, sigma=1e-9) > 1.0


def test_the_spherical_fit_recovers_direction_and_distance(antennas):
    r"""Times from ``SWF_model`` fit back to the direction and Xmax distance.

    Started half a degree off in both angles, as the reconstruction chain
    starts it from the plane-wave result.  About eight seconds: the fit is a
    differential evolution over a pure-Python loss.
    """
    import grand.analysis.constants as cons
    from grand.analysis.fitting.spherical import SWF_model, recons_swf

    zenith, azimuth, distance = np.deg2rad(62.0), np.deg2rad(137.0), 25000.0
    emitted = -distance / cons.c_light
    times = SWF_model(zenith, azimuth, distance, emitted, antennas)

    theta, phi, r, t_s = recons_swf(zenith + np.deg2rad(0.5),
                                    azimuth - np.deg2rad(0.5),
                                    times, antennas, maxiter=300)

    assert np.degrees(theta) == pytest.approx(62.0, abs=1e-3)
    assert np.degrees(phi) == pytest.approx(137.0, abs=1e-3)
    assert r == pytest.approx(distance, rel=1e-4)
    assert t_s == pytest.approx(emitted, rel=1e-4)


def test_the_reconstruction_uses_comes_from_angles():
    r"""The shower axis points *away* from where the angles say it came from.

    A shower from zenith 0 travels straight down; one from azimuth 0 travels
    towards -x.  The same convention as the rest of GRANDlib.
    """
    from grand.analysis.coords.array_shower import shower_direction_vector

    assert np.allclose(shower_direction_vector(0.0, 0.0), [0.0, 0.0, -1.0])
    k = shower_direction_vector(np.pi / 2, 0.0)
    assert np.allclose(k, [-1.0, 0.0, 0.0])


def test_the_voltage_energy_proxy():
    r"""The linear proxy of arXiv:2507.04324, and its clamp at zero."""
    from grand.analysis.energy_reco.voltage import recons_energy_from_voltage

    a, b = 1.96e7, 7.90e6
    amplitude, sin_alpha = 3.0e7, 0.8
    expected = (amplitude / sin_alpha - b) / a * 1e18
    assert recons_energy_from_voltage(amplitude, sin_alpha) == pytest.approx(expected)
    assert recons_energy_from_voltage(1.0, 0.8) == 0.0


def test_the_energy_proxy_does_not_take_arrays_yet():
    r"""Pinned defect: documented as array-like, raises on an array.

    ``max(energy, 0.0)`` is the builtin, which cannot compare an array.  When
    this is fixed (``np.maximum``), this test fails and should be replaced by
    one that checks the array result.
    """
    from grand.analysis.energy_reco.voltage import recons_energy_from_voltage

    with pytest.raises(ValueError):
        recons_energy_from_voltage(np.array([3.0e7, 1.0]), 0.8)
