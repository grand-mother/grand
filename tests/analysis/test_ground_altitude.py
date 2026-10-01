# -*- coding: utf-8 -*-
r"""#252: the fits' frame and simulation files.

The fits take antenna heights above sea level and place the source relative
to ``groundAltitude`` (1231 m, GP13); ``TRun.du_xyz`` is relative to the run's
``origin_geoid``.  ``antenna_positions_from_run`` converts one to the other.
"""

import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000" / "run_1_L0_0000.root"

pytestmark = pytest.mark.skipif(not RUN.exists(), reason="the committed sample is not present")


def test_positions_and_ground_come_from_the_run():
    from grand.analysis.geom import antenna_positions_from_run
    from grand.dataio import TRun

    with TRun(str(RUN)) as run:
        run.get_entry(0)
        xyz = np.asarray(run.du_xyz, dtype=float)
        origin = np.ravel(np.asarray(run.origin_geoid, dtype=float))
        Xants, ground = antenna_positions_from_run(run)
    assert ground == pytest.approx(origin[2]) and ground > 1000
    assert Xants.shape == xyz.shape
    np.testing.assert_allclose(Xants[:, :2], xyz[:, :2])
    np.testing.assert_allclose(Xants[:, 2], xyz[:, 2] + origin[2])


def test_the_converted_frame_matches_the_run_frame():
    # The model in the converted frame equals the model with the antennas and
    # the ground shifted together; with du_xyz and the default ground it does not
    from grand.analysis.fitting.spherical import SWF_model
    from grand.analysis.geom import antenna_positions_from_run
    from grand.dataio import TRun

    with TRun(str(RUN)) as run:
        run.get_entry(0)
        Xants, ground = antenna_positions_from_run(run)
        xyz = np.asarray(run.du_xyz, dtype=float)
    theta, phi, r, t = np.radians(60.0), np.radians(30.0), 20e3, 0.0
    right = SWF_model(theta, phi, r, t, Xants, groundAltitude=ground)
    wrong = SWF_model(theta, phi, r, t, xyz)
    assert np.ptp(right - wrong) > 10e-9          # tens of ns of differential error
