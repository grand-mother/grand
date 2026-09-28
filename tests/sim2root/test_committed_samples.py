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
