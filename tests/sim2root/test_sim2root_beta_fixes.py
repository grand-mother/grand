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


def _sim2root_fails(tmp_path, *args):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    done = subprocess.run([sys.executable, str(SIM2ROOT), *args, "-sl", "GP300", "-o", str(out)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert done.returncode != 0
    assert not list(out.rglob("*.root")), "files were written before the check"
    return done.stderr


@pytest.mark.parametrize("options, message", [
    (("--target_duration_us", "0.1"), "ends"),
    (("--trigger_time_ns", "3000", "--target_duration_us", "2"), "shorter than"),
    (("--target_duration_us", "0"), "must be > 0"),
    (("--trigger_time_ns", "0"), "must be > 0"),
])
def test_window_options_are_checked_before_writing(tmp_path, options, message):
    r"""#222: impossible windows wrote all-zero traces, or failed with a bare error."""
    raw = shutil.copy(ZHAIRES / (RUN_13790 + ".rawroot"), tmp_path)
    stderr = _sim2root_fails(tmp_path, pathlib.Path(raw).name, *options)
    assert "GRANDlib: sim2root:" in stderr and message in stderr, stderr[-2000:]


def test_events_with_different_windows_need_one_window(tmp_path):
    r"""#222: a run of events with different windows stored the first event's window."""
    from grand.dataio import TEfield, TRunEfieldSim

    raws = [pathlib.Path(shutil.copy(p, tmp_path)).name for p in sorted(ZHAIRES.glob("*.rawroot"))]
    assert "differs from the run's" in _sim2root_fails(tmp_path, *raws)
    assert "differs from the run's" in _sim2root_fails(tmp_path, *raws, "--trigger_time_ns", "500")

    out = _run_sim2root(tmp_path, *raws, "--trigger_time_ns", "500", "--target_duration_us", "2")
    run = TRunEfieldSim(str(next(out.glob("*/runefieldsim_*.root"))))
    run.get_entry(0)
    assert (run.t_pre, run.t_post) == (500, 1500)
    run.stop_using()
    efield = TEfield(str(next(out.glob("*/efield_*.root"))))
    for entry in range(efield.get_entries()):
        efield.get_entry(entry)
        assert np.asarray(efield.trace).shape[-1] == 4000
        assert set(np.asarray(efield.trigger_position)) == {1000}
    efield.stop_using()


def test_the_gamma_and_hadron_profiles_reach_tshowersim(tmp_path):
    r"""#202: assigned under names TShowerSim does not have, they were stored nowhere."""
    from grand.dataio import TShowerSim

    raw = shutil.copy(ZHAIRES / (RUN_13790 + ".rawroot"), tmp_path)
    out = _run_sim2root(tmp_path, pathlib.Path(raw).name)
    sim = TShowerSim(str(next(out.glob("*/showersim_*.root"))))
    sim.get_entry(0)
    assert len(sim.long_pd_gamma) == len(sim.long_pd_depth) > 0
    assert max(sim.long_pd_gamma) > 0
    assert len(sim.long_pd_hadron) == len(sim.long_pd_depth)
    sim.stop_using()
