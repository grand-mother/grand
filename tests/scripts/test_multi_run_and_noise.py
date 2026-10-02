# -*- coding: utf-8 -*-
r"""#241: folders holding several runs, and the measured-noise folder.

A folder with two runs crashed ``convert_efield2voltage`` (its run chain had
no usable index); fewer noise traces than antennas reused them silently,
correlating the antennas' noise; and a simulated ADC file or too short
traces as noise failed with an ``IndexError`` or an ``assert``.
"""

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMON = ROOT / "sim2root" / "Common"
RUN0 = COMMON / "sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS_0000"
RUN1 = COMMON / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
NOISE = next((COMMON / "LongNoiseTraces").glob("*.root"), None)

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "detector").exists() or not RUN0.is_dir()
                                or not RUN1.is_dir() or NOISE is None,
                                reason="needs the data model, the RUN0/RUN1 samples and the noise traces")


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=900)


@pytest.fixture
def two_runs(tmp_path):
    folder = tmp_path / "two"
    folder.mkdir()
    for run in (RUN0, RUN1):
        for path in run.glob("*_L0_*"):
            if not path.name.startswith(("voltage_", "adc_")):
                shutil.copy(path, folder)
    return folder


def test_a_folder_with_two_runs_finds_both(two_runs):
    from grand.dataio import DataDirectory

    directory = DataDirectory(str(two_runs))
    assert directory.trun.get_run(0) > 0 and int(directory.trun.run_number) == 0
    assert directory.trun.get_run(1) > 0 and int(directory.trun.run_number) == 1


def test_efield2voltage_converts_a_folder_with_two_runs(two_runs):
    from grand.dataio import TVoltage

    done = _run("convert_efield2voltage.py", two_runs, "--seed", "1")
    assert done.returncode == 0, done.stderr[-2000:]
    runs = set()
    for voltage in two_runs.glob("voltage_*.root"):
        with TVoltage(str(voltage)) as tree:
            runs |= {run for _, run in tree.get_list_of_events()}
    assert runs == {0, 1}


def _noise_file(path, entries, samples=None):
    r"""Writes `entries` entries of the committed noise file into `path`, cut to `samples`."""
    from grand.dataio import TADC

    src, dst = TADC(str(NOISE)), TADC(str(path))
    for entry in range(entries):
        src.get_entry(entry)
        trace = np.asarray(src.trace_ch)
        dst.copy_contents(src)
        if samples is not None:
            dst.trace_ch = trace[:, :, :samples].tolist()
            dst.adc_samples_count_ch = [[samples] * 3]
        dst.fill()
    dst.write()
    src.stop_using()
    dst.stop_using()


@pytest.fixture
def voltage_folder(tmp_path):
    folder = tmp_path / "sample"
    shutil.copytree(RUN1, folder)
    for path in folder.glob("adc_*"):
        path.unlink()
    return folder


def test_too_few_noise_traces_warn(voltage_folder, tmp_path):
    noise = tmp_path / "noise"
    noise.mkdir()
    _noise_file(noise / "few.root", 3)
    done = _run("convert_voltage2adc.py", voltage_folder, "--add_noise_from", noise, "-s", "1")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "noise traces for" in done.stderr and "reused" in done.stderr


def test_a_simulated_adc_file_is_refused_as_noise(voltage_folder, tmp_path):
    noise = tmp_path / "noise"
    noise.mkdir()
    shutil.copy(next(RUN0.glob("adc_*.root")), noise)
    done = _run("convert_voltage2adc.py", voltage_folder, "--add_noise_from", noise, "-s", "1")
    assert done.returncode != 0
    assert "GRANDlib" in done.stderr and "adc_" in done.stderr and "IndexError" not in done.stderr
    assert not list(voltage_folder.glob("adc_*"))


def test_short_noise_traces_are_refused(voltage_folder, tmp_path):
    noise = tmp_path / "noise"
    noise.mkdir()
    _noise_file(noise / "short.root", 50, samples=100)
    done = _run("convert_voltage2adc.py", voltage_folder, "--add_noise_from", noise, "-s", "1")
    assert done.returncode != 0
    assert "short.root" in done.stderr and "100 samples" in done.stderr
    assert "AssertionError" not in done.stderr
