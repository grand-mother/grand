# -*- coding: utf-8 -*-
r"""#253: the effective-length lookup was 1 degree off for every negative azimuth.

The tables' azimuth grid runs 0-360 degrees inclusive (361 points: 360 repeats
0), and the wrap used the 361 points, so a direction at -30 degrees read the
330-331 interval as 331-332.  A table whose value is its own azimuth shows
which azimuth was read.
"""

import dataclasses

import numpy as np
import pytest

from grand import grand_add_path_data

MODEL = grand_add_path_data("detector/Light_GP300Antenna_EWarm_leff.npz")


@pytest.fixture(scope="module")
def model():
    import os

    if not os.path.exists(MODEL):
        pytest.skip("needs the GRAND data model (data/detector)")
    from grand.sim.detector.antenna_model import tabulated_antenna_model

    return tabulated_antenna_model(MODEL)


def _read_azimuth(model, azimuth_deg, zenith_deg=40.0):
    from grand import LTP, ECEF, CartesianRepresentation
    from grand.basis.type_trace import ElectricField
    from grand.sim.detector.process_ant import AntennaProcessing

    table = np.broadcast_to(model.phi[None, :, None].astype(float),
                            model.leff_theta_reim.shape).astype(model.leff_theta_reim.dtype)
    synthetic = dataclasses.replace(model, leff_theta_reim=table,
                                    leff_phi_reim=np.zeros_like(model.leff_phi_reim))
    loc = ECEF(x=-202152.62, y=4968285.91, z=3981091.07)
    frame = LTP(x=0, y=0, z=0, location=loc, orientation="NWU", declination=0.0)
    n = 1024
    field = ElectricField(np.arange(n) * 0.5e-9,
                          CartesianRepresentation(x=np.zeros(n), y=np.zeros(n), z=np.zeros(n)),
                          frame=frame)
    t, p, r = np.deg2rad(zenith_deg), np.deg2rad(azimuth_deg), 20000.0
    xmax = LTP(x=r * np.sin(t) * np.cos(p), y=r * np.sin(t) * np.sin(p), z=r * np.cos(t), frame=frame)
    antenna = AntennaProcessing(model_leff=synthetic, pos=frame)
    antenna.effective_length(xmax, field, frame)
    in_band = np.abs(antenna.l_t)[np.abs(antenna.l_t) > 0]
    return float(np.median(in_band))


@pytest.mark.parametrize("azimuth, expected", [(30.5, 30.5), (-29.5, 330.5), (-150.25, 209.75),
                                               (-0.5, 359.5)])
def test_negative_azimuths_read_their_own_table_column(model, azimuth, expected):
    assert _read_azimuth(model, azimuth) == pytest.approx(expected, abs=1e-3)
