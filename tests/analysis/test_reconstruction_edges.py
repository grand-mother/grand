"""Edge cases of the direction reconstruction and its documentation (#216).

A horizontal shower gave NaN from the plane-wave fit, an azimuth of 0 could
come back as 2*pi, and several docstrings left out what the angles mean.
"""
import pathlib

import numpy as np
import pytest

pytest.importorskip('iminuit', reason='grand.analysis needs iminuit')

from grand.analysis.fitting import plane_wave, spherical  # noqa: E402
from grand.analysis.cramer_rao_bounds import cramer_rao  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _array(flat=False):
    rng = np.random.default_rng(3)
    xants = rng.uniform(-2000, 2000, size=(20, 3))
    xants[:, 2] = 1200.0 if flat else 1200.0 + rng.uniform(-50, 50, 20)
    return xants


def _fit(theta_deg, phi_deg, flat=False):
    xants = _array(flat)
    params = np.deg2rad([theta_deg, phi_deg])
    tants = plane_wave.PWF_model(params, xants)
    return np.rad2deg(plane_wave.PWF_semianalytical(xants, tants))


# A flat array takes the degenerate branch, where both failures showed
@pytest.mark.parametrize("flat", [True, False])
def test_horizontal_shower_is_finite(flat):
    theta, phi = _fit(90.0, 0.0, flat)
    assert np.isfinite([theta, phi]).all()
    assert theta == pytest.approx(90.0, abs=0.1)
    assert min(phi, 360.0 - phi) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("flat", [True, False])
@pytest.mark.parametrize("phi", [0.0, -1e-12, 45.0, 180.0, 359.9999])
def test_azimuth_in_range(phi, flat):
    theta_fit, phi_fit = _fit(60.0, phi, flat)
    assert 0.0 <= phi_fit < 360.0
    assert theta_fit == pytest.approx(60.0, abs=0.1)
    d = abs(phi_fit - phi % 360.0)
    assert min(d, 360.0 - d) < 0.1


def test_vertical_flat_array():
    theta, phi = _fit(0.0, 0.0, flat=True)
    assert np.isfinite([theta, phi]).all()
    assert theta == pytest.approx(0.0, abs=0.1)
    assert 0.0 <= phi < 360.0


def test_docstrings_say_what_the_angles_mean():
    doc = plane_wave.PWF_semianalytical.__doc__
    assert "sigma" in doc and "comes from" in doc and "2*pi" in doc
    assert "comes from" in spherical.SWF_loss.__doc__
    assert "width" in cramer_rao.__doc__ or any(
        "delta_omega" in (f.__doc__ or "") for f in vars(cramer_rao).values()
        if callable(f))


def test_event_viewer_labels():
    pytest.importorskip("holoviews")
    pytest.importorskip("pandas")
    sample = (ROOT / "sim2root" / "Common"
              / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
    if not sample.is_dir():
        pytest.skip("needs the RUN1 sample")
    import sys
    sys.path.insert(0, str(ROOT / "examples" / "eventviewer"))
    try:
        import event_viewer_to_root as viewer
    finally:
        sys.path.pop(0)
    assert viewer.EventViewer.time_label == "Sample"
    v = viewer.EventViewer(str(sample))
    v.get_geometry()
    v.get_data()
    assert v.time_label == "Sample (2 ns each)"
    assert "'Time Bins'" not in (ROOT / "examples" / "eventviewer"
                               / "event_viewer_to_root.py").read_text()
