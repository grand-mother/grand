# -*- coding: utf-8 -*-
r"""#225: unknown values from a ZHAireS simulation are NaN, and a missing event
time has one documented fallback.

An Xmax the ``.sry`` does not give was stored as -1 g/cm2 and -1000 m, and
became a position below ground; a missing event time became 200854920 in the
converter and 200854852 in sim2root (May 1976), while ``unix_date`` and the
run's event times said 2022-10-26.
"""

import datetime
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tests.sim2root.test_sim2root_beta_fixes import RUN_13790, SIM2ROOT, ZHAIRES, _convert

pytestmark = pytest.mark.skipif(not (ZHAIRES / RUN_13790).is_dir(), reason="the committed samples are not present")


def _env():
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _sim2root(tmp_path, raw):
    out = tmp_path / "out"
    out.mkdir()
    done = subprocess.run([sys.executable, str(SIM2ROOT), str(raw), "-sl", "GP300", "-o", str(out)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=900, env=_env())
    assert done.returncode == 0, done.stderr[-2000:]
    return out, done


def test_an_unknown_xmax_is_nan(tmp_path):
    from grand.dataio import TShower

    folder = tmp_path / RUN_13790
    shutil.copytree(ZHAIRES / RUN_13790, folder)
    sry = folder / (RUN_13790 + ".sry")
    lines = sry.read_text(errors="replace").splitlines(keepends=True)
    sry.write_text("".join(line for line in lines
                           if "Pos. Max.:" not in line and "Sl. depth of max." not in line))
    raw = tmp_path / "noxmax.rawroot"
    done = subprocess.run([sys.executable, str(ZHAIRES / "ZHAireSRawToRawROOT.py"), "./" + RUN_13790,
                           "standard", "1", "5", str(raw)],
                          cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "stored as NaN" in done.stderr

    out, _ = _sim2root(tmp_path, raw)
    (path,) = out.rglob("shower_*.root")
    with TShower(str(path)) as shower:
        shower.get_entry(0)
        assert np.isnan(shower.xmax_grams)
        assert np.isnan(np.asarray(shower.xmax_pos, dtype=float)).all()


def test_a_missing_event_time_is_the_simulation_date_everywhere(tmp_path):
    from grand.dataio import TEfield, TRun, TShower

    raw = _convert(tmp_path, (5,), "one.rawroot")     # the sample's EventUnixTime is 0
    out, done = _sim2root(tmp_path, raw)
    assert "gives no event time" in done.stderr + done.stdout

    (shower_path,) = out.rglob("shower_*.root")
    (efield_path,) = out.rglob("efield_*.root")
    (run_path,) = out.rglob("run_*.root")
    with TShower(str(shower_path)) as shower, TEfield(str(efield_path)) as efield, \
            TRun(str(run_path)) as run:
        shower.get_entry(0)
        efield.get_entry(0)
        run.get_entry(0)
        seconds = int(shower.core_time_s)
        day = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).date()
        assert day == datetime.date(2022, 10, 26)
        assert int(run.first_event_time) == seconds
        du_seconds = np.asarray(efield.du_seconds, dtype=np.int64)
        assert (np.abs(du_seconds - seconds) <= 1).all()
