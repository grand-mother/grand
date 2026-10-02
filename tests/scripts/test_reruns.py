# -*- coding: utf-8 -*-
r"""#240: re-running the conversion scripts on one folder.

convert_voltage2adc.py deleted its earlier output, then failed with
NotUniqueEvent and left none; convert_efield2voltage.py failed the same way on
a re-run, and a run that failed late left a stub that broke every later run.
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


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=900)


def _events(path, cls):
    tree = cls(str(path))
    events = sorted(tree.get_list_of_events())
    level = tree.analysis_level
    tree.stop_using()
    return events, level


def test_conversions_can_be_run_again(tmp_path):
    from grand.dataio import TADC, TVoltage

    data = tmp_path / "d"
    shutil.copytree(SAMPLE, data)
    for old in data.glob("adc_*"):
        old.unlink()

    done = _run("convert_efield2voltage.py", data, "--add_jitter_ns", "-5")
    assert done.returncode != 0
    assert not list(data.glob("voltage_*_L1_*")) and not list(data.glob(".*partial*"))

    for attempt in (1, 2):
        done = _run("convert_efield2voltage.py", data, "--seed", "1")
        assert done.returncode == 0, (attempt, done.stderr[-1500:])
        done = _run("convert_voltage2adc.py", data, "-s", "1")
        assert done.returncode == 0, (attempt, done.stderr[-1500:])
        voltage = next(data.glob("voltage_*_L1_*"))
        adc = next(data.glob("adc_*"))
        assert _events(voltage, TVoltage) == ([(1618, 1), (13790, 1)], 1)
        assert _events(adc, TADC)[0] == [(1618, 1), (13790, 1)]
    assert not list(data.glob(".*partial*"))


def test_a_file_without_trees_is_skipped(tmp_path):
    import ROOT as root

    from grand.dataio import DataDirectory

    data = tmp_path / "d"
    shutil.copytree(SAMPLE, data)
    stub = root.TFile(str(data / "adc_9-9_L2_0000.root"), "recreate")
    stub.Close()
    d = DataDirectory(str(data))
    assert d.tadc.get_number_of_entries() == 2
    d.close()
