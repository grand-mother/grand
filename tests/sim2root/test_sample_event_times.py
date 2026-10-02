# -*- coding: utf-8 -*-
r"""Which committed samples still carry the May 1976 event time, and why (#225).

A simulation gives no event time (``EventUnixTime: 0``).  The ZHAireS
converter used to store 200854920 in its place and ``sim2root.py`` 200854852,
both in May 1976, while ``unix_date`` said 2022-10-26.  PR #291 replaced both
with one fallback, the simulation date, applied by ``sim2root.py``; the
converter now passes the 0 through.

* The two ``.rawroot`` samples under ``sim2root/ZHAireSRawRoot/`` are
  regenerated with today's converter; that changes ``trawmeta.unix_second``
  (200854920 to 0) and nothing else.  The first two tests fail on the old
  files.

* The four sample folders under ``sim2root/Common/`` are kept, and keep the
  1976 time.  Two are the 2024 fixtures that
  ``tests/dataio/test_backward_compatibility.py`` reads as they were written.
  The two ``RUN1`` folders cannot be regenerated for the time alone: today's
  pipeline also changes their Xmax frame (kept on purpose, see
  ``tests/sim2root/test_xmax_frame.py``), ``direction``, ``energy_em``,
  ``xmax_pos``, the run metadata, the event order and the file names, which
  the tests that read them pin.  The last test lists the folders so that a
  regenerated or added sample has to update the list, and
  ``docs/source/known_issues.rst`` says the same.
"""

import datetime
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZHAIRES = ROOT / "sim2root" / "ZHAireSRawRoot"
COMMON = ROOT / "sim2root" / "Common"
SIM2ROOT = COMMON / "sim2root.py"
RAWROOTS = sorted(ZHAIRES.glob("*.rawroot"))

#: The simulation date of both ZHAireS samples, 2022-10-26, midnight UTC.
SIMULATION_DATE = 1666742400

#: May 1976: the old converter fallback, and its value one second earlier
#: once a negative t0 is added to it, then the old sim2root fallback (#225).
OLD_TIMES = (200854920, 200854919, 200854852)

#: Sample folders under sim2root/Common that keep the old time, with the reason.
KEPT_WITH_1976_TIME = {
    "sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000":
        "2024 fixture read as written by tests/dataio/test_backward_compatibility.py",
    "sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS_0000":
        "2024 fixture read as written by tests/dataio/test_backward_compatibility.py",
    "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000":
        "regenerating also changes the Xmax frame, direction, energy_em and run metadata",
    "sim_Xiaodushan_20221026_000000_RUN1_CD_ADCNoise_0000":
        "regenerating also changes the Xmax frame, direction, energy_em and run metadata",
}


def _unix_seconds(path, tree, branch):
    r"""All values of `branch` in `tree` of the ROOT file `path`, flattened."""
    import uproot

    with uproot.open(path) as f:
        values = f[tree][branch].array(library="np")
    return np.concatenate([np.ravel(np.asarray(v, dtype=np.int64)) for v in values])


@pytest.mark.skipif(len(RAWROOTS) != 2, reason="the committed samples are not present")
@pytest.mark.parametrize("raw", RAWROOTS, ids=lambda p: p.stem.rsplit("_", 1)[-1])
def test_the_rawroot_samples_store_no_event_time(raw):
    r"""``unix_second`` is 0, as today's converter writes it, not 200854920."""
    assert _unix_seconds(raw, "trawmeta", "unix_second").tolist() == [0]


@pytest.mark.skipif(len(RAWROOTS) != 2, reason="the committed samples are not present")
def test_sim2root_dates_the_rawroot_samples_to_the_simulation_day(tmp_path):
    r"""Converted, both events and every unit are timed 2022-10-26, not 1976."""
    for raw in RAWROOTS:
        shutil.copy(raw, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, str(SIM2ROOT)] + [p.name for p in RAWROOTS]
        + ["-se", "1", "-sl", "GP300", "-o", str(out),
           "--trigger_time_ns", "500", "--target_duration_us", "2"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]

    (shower,) = out.rglob("shower_*.root")
    (efield,) = out.rglob("efield_*.root")
    core = _unix_seconds(shower, "tshower", "core_time_s")
    assert core.tolist() == [SIMULATION_DATE, SIMULATION_DATE]
    day = datetime.datetime.fromtimestamp(int(core[0]), datetime.timezone.utc).date()
    assert day == datetime.date(2022, 10, 26)
    # A negative t0 puts a unit up to one second before the event time.
    du = _unix_seconds(efield, "tefield", "du_seconds")
    assert ((du >= SIMULATION_DATE - 1) & (du <= SIMULATION_DATE)).all()


def _carries_1976_time(folder):
    shower = next(folder.glob("shower_*_L0_*.root"))
    return bool(np.isin(_unix_seconds(shower, "tshower", "core_time_s"), OLD_TIMES).any())


def test_which_common_samples_keep_the_1976_time():
    r"""Exactly the folders listed above, so a regenerated one updates the list."""
    folders = sorted(p for p in COMMON.glob("sim_*") if p.is_dir())
    if not folders:
        pytest.skip("the committed samples are not present")
    carrying = {p.name for p in folders if _carries_1976_time(p)}
    assert carrying == set(KEPT_WITH_1976_TIME)
