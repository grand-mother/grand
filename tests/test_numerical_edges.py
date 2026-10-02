"""Numerical edges of #289: one-sided guards, an ignored sigma, trigger
shapes, the analysis ADC helper, downsampling, coordinates and float32."""
import types
import warnings

import numpy as np
import pytest

from grand.basis.validate import GRANDlibWarning


def test_position_guard_is_two_sided():
    from grand.sim.efield2voltage import Efield2Voltage

    for position in ([-3e7, 0, 0], [0, 3e7, 0], [0, 0, np.nan]):
        stub = types.SimpleNamespace(du_pos=np.array([position]))
        with pytest.raises(ValueError, match="GRANDlib: Efield2Voltage.get_leff: the position of DU index 0"):
            Efield2Voltage.get_leff(stub, 0)


def test_swf_uses_sigma(monkeypatch):
    pytest.importorskip("iminuit")
    from grand.analysis.fitting import spherical

    rng = np.random.default_rng(1)
    xants = rng.uniform(-1000, 1000, (6, 3))
    tants = rng.uniform(0, 1e-6, 6)
    plain = spherical.SWF_loss(1.0, 0.5, 1e4, -1e-5, xants, tants)
    sigma = np.full(6, 2e-9)
    assert spherical.SWF_loss(1.0, 0.5, 1e4, -1e-5, xants, tants, sigma=2e-9) == pytest.approx(
        plain / (spherical.cons.c_light * 2e-9)**2)
    weighted = spherical.SWF_loss(1.0, 0.5, 1e4, -1e-5, xants, tants, sigma=sigma)
    assert isinstance(weighted, float) and weighted == pytest.approx(
        spherical.SWF_loss(1.0, 0.5, 1e4, -1e-5, xants, tants, sigma=np.diag(sigma**2)))

    seen = []
    real = spherical.SWF_loss
    monkeypatch.setattr(spherical, "SWF_loss",
                        lambda *a, sigma=None, **k: seen.append(sigma) or real(*a, sigma=sigma, **k))
    spherical.recons_swf(1.0, 0.5, tants, xants, sigma=sigma, maxiter=1)
    assert seen and all(s is not None and np.array_equal(s, sigma) for s in seen)


def test_trigger_shape_and_period():
    from grand.sim.detector.trigger import t1_du_triggers

    trace = np.random.default_rng(0).normal(0, 30, (3, 1024))
    with pytest.raises(ValueError, match=r"shape \(N_du, N_channels, N_samples\), got \(3, 1024\)"):
        t1_du_triggers(trace)
    for period in (0, 1):
        with pytest.raises(ValueError, match="t_period must be at least 2"):
            t1_du_triggers(trace[None], {"t_period": period})


def test_analysis_adc_truncates_and_saturates():
    from grand.analysis.signals.extraction import convert_voltage_to_ADC

    trace = np.array([[1e12, -1e12, 150.0, -150.0], [0.0, 1.0, 2.0, 3.0]])
    counts = convert_voltage_to_ADC(trace, [0])
    assert counts[0].tolist() == [8192, -8192, 1, -1]
    assert counts[1].tolist() == [0.0, 1.0, 2.0, 3.0]          # not selected: unchanged
    with pytest.raises(ValueError, match="NaN or infinity"):
        convert_voltage_to_ADC(np.full((1, 4), np.nan), [0])


def test_downsample_rates_and_odd_lengths():
    from grand import ADC

    adc = ADC()
    trace = np.zeros((1, 3, 999))
    for rate in (0, np.nan, -500):
        with pytest.raises((ValueError, TypeError), match="input_sampling_rate_mhz"):
            adc.downsample(trace, rate)
    assert adc.downsample(trace, 1000).shape == (1, 3, 500)


def test_coordinate_edges():
    from grand.geo.coordinates import LTP, Geodetic, Horizontal, geoid_undulation

    site = Geodetic(latitude=40.0, longitude=90.0, height=0.0)
    direction = Horizontal(azimuth=0.0, elevation=0.0, norm=1.0)
    with pytest.raises(TypeError) as error:
        LTP(direction, location=site, orientation="NWU")
    assert "or Horizontal" not in str(error.value)

    from grand.geo.coordinates import ECEF
    flat = Horizontal(ECEF(site), location=site)
    assert np.all(np.isfinite(np.asarray(flat.elevation, dtype=float)))

    with pytest.raises(ValueError, match="latitude must be within"):
        geoid_undulation(91, 0)
    with pytest.raises(ValueError, match="longitude must be within"):
        geoid_undulation(0, 361)
    assert np.isfinite(geoid_undulation(0, -200))


def test_float32_fields_refuse_overflow():
    from grand.dataio import TShower

    shower = TShower()
    with pytest.raises(ValueError, match="does not fit in float32"):
        shower.energy_primary = 1e39
    with pytest.warns(GRANDlibWarning, match="below the smallest float32"):
        shower.energy_primary = 1e-46
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shower.energy_primary = np.inf                  # inf given is inf stored
        shower.energy_primary = 1e30
