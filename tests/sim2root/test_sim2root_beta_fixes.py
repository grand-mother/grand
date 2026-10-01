# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to ``sim2root.py``.

Each test names its GitHub issue and fails without the fix.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZHAIRES = ROOT / "sim2root" / "ZHAireSRawRoot"
SIM2ROOT = ROOT / "sim2root" / "Common" / "sim2root.py"
RUN_13790 = "GP300_Xi_Sib_Proton_3.87_79.4_310.0_13790"

pytestmark = pytest.mark.skipif(not (ZHAIRES / RUN_13790).is_dir(), reason="the committed samples are not present")


def _convert(tmp_path, event_ids, out_name):
    r"""Converts the committed 13790 shower into ``out_name``, once per event id."""
    shutil.copytree(ZHAIRES / RUN_13790, tmp_path / RUN_13790, dirs_exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for event_id in event_ids:
        done = subprocess.run(
            [sys.executable, str(ZHAIRES / "ZHAireSRawToRawROOT.py"), "./" + RUN_13790, "standard", "1",
             str(event_id), str(tmp_path / out_name)],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
        assert "Error" not in done.stderr, done.stderr[-2000:]
    return tmp_path / out_name


def _run_sim2root(tmp_path, *args):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    done = subprocess.run([sys.executable, str(SIM2ROOT), *args, "-sl", "GP300", "-o", str(out)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    return out


def _geoid_from_xyz(run):
    from grand.geo.coordinates import Geodetic, GRANDCS

    origin = np.asarray(run.origin_geoid, dtype=float)
    xyz = np.asarray(run.du_xyz, dtype=float)
    site = Geodetic(latitude=origin[0], longitude=origin[1], height=origin[2])
    g = Geodetic(GRANDCS(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], location=site))
    return np.c_[np.ravel(g.latitude), np.ravel(g.longitude), np.ravel(g.height)]


def test_du_geoid_matches_du_xyz_when_a_file_holds_several_events(tmp_path):
    r"""#220: several events with the same antennas in one file gave du_geoid kilometres off."""
    from grand.dataio import TRun

    raw = _convert(tmp_path, (5, 6), "same.rawroot")
    out = _run_sim2root(tmp_path, raw.name)
    run = TRun(str(next(out.glob("*/run_*_L0_*.root"))))
    run.get_entry(0)
    geoid = np.asarray(run.du_geoid, dtype=float)
    assert geoid.shape == (len(run.du_id), 3)
    assert np.allclose(geoid[:, :2], _geoid_from_xyz(run)[:, :2], atol=1e-5)
    assert np.allclose(geoid[:, 2], _geoid_from_xyz(run)[:, 2], atol=0.1)
    run.stop_using()


def test_star_shape_runs_have_their_own_du_geoid_and_first_event(tmp_path):
    r"""#220: with -ss, du_geoid was empty and every run kept the first run's first event."""
    from grand.dataio import TRun

    raws = [shutil.copy(p, tmp_path) for p in sorted(ZHAIRES.glob("*.rawroot"))]
    out = _run_sim2root(tmp_path, *[pathlib.Path(p).name for p in raws], "-ss")
    run = TRun(str(next(out.glob("*/run_*_L0_*.root"))))
    assert run.get_number_of_entries() == 2
    first_events = []
    for entry in range(2):
        run.get_entry(entry)
        assert len(run.du_geoid) == len(run.du_id) == len(run.du_tilt) == len(run.t_bin_size)
        assert np.allclose(np.asarray(run.du_geoid, dtype=float)[:, :2], _geoid_from_xyz(run)[:, :2], atol=1e-5)
        first_events.append(int(run.first_event))
    assert len(set(first_events)) == 2
    run.stop_using()
