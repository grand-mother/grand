# -*- coding: utf-8 -*-
r"""Conversion scripts: missing output folders (#182) and a file for a folder (#180)."""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZHAIRES = sorted((ROOT / "sim2root" / "ZHAireSRawRoot").glob("*.rawroot"))

pytestmark = pytest.mark.skipif(not ZHAIRES, reason="the committed samples are not present")


def _run(*argv, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    work = tmp_path_factory.mktemp("folders")
    shutil.copy(ZHAIRES[0], work)
    _run(ROOT / "sim2root" / "Common" / "sim2root.py", ZHAIRES[0].name, "-sl", "GP300", cwd=work)
    (folder,) = work.glob("sim_*")
    return folder


def test_a_new_output_folder_is_created(simulation, tmp_path):
    target = tmp_path / "new" / "sub"
    _run(ROOT / "scripts" / "convert_efield2voltage.py", simulation, "--no_noise", "-od", target, cwd=tmp_path)
    assert list(target.glob("voltage_*_L0_*.root"))
    target = tmp_path / "other" / "sub"
    _run(ROOT / "scripts" / "convert_efield2efield.py", simulation, "-od", target, cwd=tmp_path)
    assert list(target.glob("*.root"))


def test_voltage2adc_accepts_the_voltage_file(simulation, tmp_path):
    _run(ROOT / "scripts" / "convert_efield2voltage.py", simulation, "--no_noise", cwd=tmp_path)
    (voltage,) = simulation.glob("voltage_*_L0_*.root")
    _run(ROOT / "scripts" / "convert_voltage2adc.py", voltage, cwd=tmp_path)
    assert list(simulation.glob("adc_*_L1_*.root"))
