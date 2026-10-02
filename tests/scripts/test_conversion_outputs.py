# -*- coding: utf-8 -*-
r"""#231: where the conversion scripts write, reruns, and the level they read.

``convert_efield2efield.py -od`` wrote the run files into the input folder and
failed on a rerun (or on the committed sample, which already holds L1 files);
``convert_voltage2adc.py`` found only ``voltage_*_L0_*.root`` and put a bare
``-o`` in the current directory; ``convert_efield2voltage.py`` read the highest
efield level of a mixed folder without saying so.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "detector").exists() or not SAMPLE.is_dir(),
                                reason="needs the data model and the RUN1 sample")


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=900, cwd=cwd)


@pytest.fixture
def sample(tmp_path):
    target = tmp_path / "sample"
    shutil.copytree(SAMPLE, target)
    return target


def test_efield2efield_writes_everything_to_od_and_can_rerun(sample, tmp_path):
    for path in sample.glob("*_L1_*"):
        path.unlink()
    before = sorted(p.name for p in sample.iterdir())
    out = tmp_path / "e2e"
    for attempt in (1, 2):
        done = _run("convert_efield2efield.py", sample, "-od", out, "--seed", "3")
        assert done.returncode == 0, (attempt, done.stderr[-1500:])
    assert sorted(p.name for p in sample.iterdir()) == before, "something was written to the input"
    names = sorted(p.name for p in out.iterdir())
    assert any(n.startswith("efield_") and "_L1_" in n for n in names)
    assert any(n.startswith("run_") and "_L1_" in n for n in names)
    assert any(n.startswith("runefieldsim_") and "_L1_" in n for n in names)


def test_efield2efield_runs_on_a_folder_already_holding_level_1(sample):
    done = _run("convert_efield2efield.py", sample, "--seed", "3")
    assert done.returncode == 0, done.stderr[-1500:]


def test_efield2voltage_says_which_level_it_reads(sample, tmp_path):
    out = tmp_path / "v"
    done = _run("convert_efield2voltage.py", sample, "-od", out, "--seed", "1")
    assert done.returncode == 0, done.stderr[-1500:]
    assert "levels 0, 1: reading the highest, 1" in done.stderr
    assert "reading level 1: efield" in done.stdout + done.stderr

    out = tmp_path / "v0"
    done = _run("convert_efield2voltage.py", sample, "-od", out, "--seed", "1", "--level", "0")
    assert done.returncode == 0, done.stderr[-1500:]
    assert "reading the highest" not in done.stderr
    assert "reading level 0: efield" in done.stdout + done.stderr


def test_voltage2adc_takes_a_voltage_file_of_any_name(sample, tmp_path):
    for path in sample.glob("adc_*"):
        path.unlink()
    voltage = next(sample.glob("voltage_*_L0_*.root"))
    custom = sample / "v_custom.root"
    voltage.rename(custom)
    done = _run("convert_voltage2adc.py", custom, "-o", "adc_custom.root", "-s", "1", cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-1500:]
    assert (sample / "adc_custom.root").exists()
    assert not (tmp_path / "adc_custom.root").exists()
