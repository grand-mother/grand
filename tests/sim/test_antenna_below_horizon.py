# -*- coding: utf-8 -*-
r"""#285: directions below the antenna's horizon read the wrong table row.

The GP300 effective-length tables cover zenith 0-90 deg.  The row index was
taken modulo the table size, so 91 deg read the zenith row, 95 deg the 4 deg
row, and so on, at full strength.  Outside the table the response is now zero,
continuous with the 90 deg row.
"""

import numpy as np
import pytest

from grand import LTP, ECEF, CartesianRepresentation, grand_add_path_data
from grand.basis.type_trace import ElectricField

MODEL = grand_add_path_data("detector/Light_GP300Antenna_EWarm_leff.npz")


@pytest.fixture(scope="module")
def model():
    import os

    if not os.path.exists(MODEL):
        pytest.skip("needs the GRAND data model (data/detector)")
    from grand.sim.detector.antenna_model import tabulated_antenna_model

    return tabulated_antenna_model(MODEL)


def _leff_peak(model, zenith_deg, azimuth_deg=30.0):
    r"""Peak |Leff_theta| for a source at (zenith, azimuth) in the antenna frame."""
    from grand.sim.detector.process_ant import AntennaProcessing

    loc = ECEF(x=-202152.62, y=4968285.91, z=3981091.07)
    frame = LTP(x=0, y=0, z=0, location=loc, orientation="NWU", declination=0.0)
    n = 1024
    field = ElectricField(np.arange(n) * 0.5e-9,
                          CartesianRepresentation(x=np.zeros(n), y=np.zeros(n), z=np.zeros(n)),
                          frame=frame)
    t, p, r = np.deg2rad(zenith_deg), np.deg2rad(azimuth_deg), 20000.0
    xmax = LTP(x=r * np.sin(t) * np.cos(p), y=r * np.sin(t) * np.sin(p), z=r * np.cos(t), frame=frame)
    antenna = AntennaProcessing(model_leff=model, pos=frame)
    antenna.effective_length(xmax, field, frame)
    assert np.ravel(antenna.theta_efield)[0] == pytest.approx(zenith_deg)
    return float(np.abs(antenna.l_t).max()), float(np.abs(antenna.l_p).max())


@pytest.mark.parametrize("zenith", [90.5, 91.0, 95.0, 120.0, 170.0])
def test_no_response_below_the_horizon(model, zenith):
    assert _leff_peak(model, zenith) == (0.0, 0.0)


def test_above_the_horizon_is_unchanged(model):
    assert _leff_peak(model, 4.0)[0] > 1.0          # strong response near the zenith
    grazing = _leff_peak(model, 89.9)[0]
    assert 0.0 < grazing < 0.01                     # weak but finite at the horizon
