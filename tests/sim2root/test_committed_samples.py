# -*- coding: utf-8 -*-
r"""The example files committed under ``sim2root/ZHAireSRawRoot`` still work.

The README's walk-through converts the two ZHAireS runs to ``.rawroot`` and
then runs ``sim2root.py`` on them.  Both steps had stopped working: the
converter failed on ``import sim2root...`` when run from its own folder, as
documented, and the committed ``.rawroot`` files were from an older format,
on which ``sim2root.py`` raised ``IndexError`` at ``trawshower.energy_em[0]``.
The files were regenerated in 2026-09; these tests fail if they fall behind
the converters again.
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
RAWROOTS = sorted(ZHAIRES.glob("*.rawroot"))

pytestmark = pytest.mark.skipif(len(RAWROOTS) != 2, reason="the committed samples are not present")


def test_sim2root_converts_the_committed_samples(tmp_path):
    r"""Both samples convert; the shower tree holds both events."""
    from grand.dataio import TShower

    for raw in RAWROOTS:
        shutil.copy(raw, tmp_path)
    done = subprocess.run(
        [sys.executable, str(SIM2ROOT)] + [p.name for p in sorted(tmp_path.glob("*.rawroot"))]
        + ["-se", "1", "-sl", "GP300", "-s", "Xiaodushan"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]

    shower = TShower(str(next(tmp_path.glob("*/shower_*_L0_*.root"))))
    assert len(shower.get_list_of_events()) == 2


def test_the_converter_runs_from_its_own_folder_as_documented(tmp_path):
    r"""``python ZHAireSRawToRawROOT.py <run> standard <run id> <event> <out>``.

    Run from ``sim2root/ZHAireSRawRoot`` with no ``PYTHONPATH``, as the README
    shows; the output goes to a temporary directory.
    """
    run = "GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = tmp_path / (run + ".rawroot")
    done = subprocess.run(
        [sys.executable, "ZHAireSRawToRawROOT.py", "./" + run, "standard", "1", "1618", str(out)],
        cwd=ZHAIRES, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    assert out.is_file()


def test_the_converter_takes_the_event_number_from_the_file_name(tmp_path):
    r"""``python ZHAireSRawToRawROOT.py <run>``: the README's one-argument form.

    The event number comes from the ``.sry`` name, as text; after input
    checks were added (PR #179) storing that text raised ``TypeError``.  The
    converter exits 0 even then, so the test checks the output and stderr.
    """
    from sim2root.Common.raw_root_trees import RawShowerTree

    run = "GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618"
    shutil.copytree(ZHAIRES / run, tmp_path / run)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, str(ZHAIRES / "ZHAireSRawToRawROOT.py"), run],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900)
    assert "Error" not in done.stderr, done.stderr[-2000:]
    raw = next(tmp_path.glob("sim_*/*.rawroot"))
    shower = RawShowerTree(str(raw))
    assert shower.get_list_of_events() == [(1618, 1)]


def test_sim2root_site_options_are_numbers(tmp_path):
    r"""``-la``/``-lo``/``-al`` set the site, and 0 is a value, not "absent"."""
    from grand.dataio import TRun

    raw = next(p for p in RAWROOTS if "1618" in p.name)
    shutil.copy(raw, tmp_path)
    done = subprocess.run(
        [sys.executable, str(SIM2ROOT), raw.name, "-sl", "GP300", "-la", "41.5", "-lo", "94.0", "-al", "0"],
        cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    run = TRun(str(next(tmp_path.glob("*/run_*_L0_*.root"))))
    run.get_entry(0)
    assert np.allclose(run.origin_geoid, [41.5, 94.0, 0.0])
