# -*- coding: utf-8 -*-
r"""#248: convert_efield2efield.py.

``du_count`` was never written (0 for every event), ``-o`` with a directory
was cut to its file name and written into the input folder, and an input
with no events logged "Exiting." and went on to write empty output files.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "convert_efield2efield.py"
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="needs the committed RUN1 sample")


@pytest.fixture
def level0(tmp_path):
    target = tmp_path / "sample"
    shutil.copytree(SAMPLE, target)
    for path in target.glob("*_L1_*"):
        path.unlink()
    return target


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          text=True, timeout=600)


def test_du_count_is_written_and_o_with_a_directory_is_honoured(level0, tmp_path):
    from grand.dataio import TEfield

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = _run(level0, "-o", elsewhere / "mine.root", "--seed", "7")
    assert done.returncode == 0, done.stderr[-1500:]
    assert not (level0 / "mine.root").exists()
    out = TEfield(str(elsewhere / "mine.root"))
    for entry in range(out.get_number_of_entries()):
        out.get_entry(entry)
        assert out.du_count == len(out.du_id) > 0
    out.stop_using()


def test_an_input_without_events_writes_nothing(level0):
    from grand.dataio import TEfield

    efield = next(level0.glob("efield_*_L0_*.root"))
    efield.unlink()
    empty = TEfield(str(efield))
    empty.write()
    empty.stop_using()
    done = _run(level0)
    assert done.returncode != 0 and "has no events" in done.stderr
    assert not list(level0.glob("efield_*_L1_*.root"))
