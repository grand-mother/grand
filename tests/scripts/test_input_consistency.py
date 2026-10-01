# -*- coding: utf-8 -*-
r"""#249: the conversion scripts check that the run, e-field and shower trees
match before computing.

Each inconsistency below crashed deep inside a script (broadcast errors,
``IndexError``, ``AttributeError`` on ``None``) or, worse, was accepted
silently.  They are now refused up front, naming what is wrong.
"""

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "detector").exists() or not SAMPLE.is_dir(),
                                reason="needs the data model and the RUN1 sample")

SCRIPTS = ["convert_efield2voltage.py", "convert_efield2efield.py"]


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=900)


@pytest.fixture
def level0(tmp_path):
    folder = tmp_path / "sample"
    shutil.copytree(SAMPLE, folder)
    for path in folder.iterdir():
        if "_L1_" in path.name or path.name.startswith(("voltage_", "adc_")):
            path.unlink()
    return folder


def _rewrite(path, cls, change):
    r"""Rewrites every entry of the `cls` tree in `path`, applying `change(tree)` to each."""
    copy = path.with_name("copy_" + path.name)
    src, dst = cls(str(path)), cls(str(copy))
    for entry in range(src.get_number_of_entries()):
        src.get_entry(entry)
        dst.copy_contents(src)
        change(dst)
        dst.fill()
    dst.write()
    src.stop_using()
    dst.stop_using()
    shutil.move(str(copy), str(path))


def _run_file(folder):
    return next(folder.glob("run_*_L0_*.root"))


def _efield_file(folder):
    return next(folder.glob("efield_*_L0_*.root"))


def _assert_refused(folder, script, *phrases):
    done = _run(script, folder, "--seed", "1")
    assert done.returncode != 0, "%s accepted it:\n%s" % (script, done.stderr[-1500:])
    tail = done.stderr[-3000:]
    assert "GRANDlib" in tail, tail
    for phrase in phrases:
        assert phrase in tail, tail
    for raw in ("could not be broadcast", "IndexError", "NoneType", "KeyError"):
        assert raw not in tail, tail


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_run_number_mismatch_is_refused(level0, script):
    from grand.dataio import TRun

    def change(tree):
        tree.run_number = 5
    _rewrite(_run_file(level0), TRun, change)
    _assert_refused(level0, script, "run 1")


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_unit_missing_from_the_run_is_refused(level0, script):
    from grand.dataio import TRun

    def change(tree):
        ids = list(tree.du_id)
        ids[0] = 99999
        tree.du_id = ids
    _rewrite(_run_file(level0), TRun, change)
    _assert_refused(level0, script, "not in the run")


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_duplicate_unit_is_refused(level0, script):
    from grand.dataio import TEfield

    def change(tree):
        ids = list(tree.du_id)
        ids[1] = ids[0]
        tree.du_id = ids
    _rewrite(_efield_file(level0), TEfield, change)
    _assert_refused(level0, script, "more than once")


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("bins", ["zero", "mixed"])
def test_a_bad_sampling_time_is_refused(level0, script, bins):
    from grand.dataio import TRun

    def change(tree):
        values = np.asarray(tree.t_bin_size, dtype=float)
        if bins == "zero":
            values[:] = 0
        else:
            values[0] = values[0] * 2
        tree.t_bin_size = values.tolist()
    _rewrite(_run_file(level0), TRun, change)
    _assert_refused(level0, script, "t_bin_size")


@pytest.mark.parametrize("script", SCRIPTS + ["convert_voltage2adc.py"])
def test_a_missing_run_file_is_named(level0, script):
    if script == "convert_voltage2adc.py":
        done = _run("convert_efield2voltage.py", level0, "--seed", "1")
        assert done.returncode == 0, done.stderr[-1500:]
    _run_file(level0).unlink()
    _assert_refused(level0, script, "run file")


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("remove", ["shower_", "efield_"])
def test_a_missing_shower_or_efield_file_is_named(level0, script, remove):
    for path in level0.glob(remove + "*"):
        path.unlink()
    _assert_refused(level0, script, remove.rstrip("_"))


def test_a_missing_folder_is_named(tmp_path):
    done = _run("convert_efield2voltage.py", tmp_path / "nowhere")
    assert done.returncode != 0 and "GRANDlib" in done.stderr and "nowhere" in done.stderr
