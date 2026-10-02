# -*- coding: utf-8 -*-
r"""#244: extract_events.py.

``-ow`` removed the whole target directory (the current one for "."); a repeated
line aborted the job and left files without a tree; a missing event exited 0.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "extract_events.py"
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="needs the committed RUN1 sample")


def _run(cwd, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=600)


def test_extract_keeps_other_files_and_writes_a_readable_target(tmp_path):
    from grand.dataio import DataDirectory

    events = tmp_path / "list.txt"
    events.write_text("# dir,run,event\n%s,1,13790\n\n%s,1,13790\n%s,1,1618\n" % ((SAMPLE,) * 3))
    target = tmp_path / "keep"
    target.mkdir()
    (target / "important.txt").write_text("x")

    done = _run(tmp_path, "-ow", "list.txt", "keep")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "repeats" in done.stdout
    done = _run(tmp_path, "-ow", "list.txt", "keep")         # again, replacing
    assert done.returncode == 0, done.stderr[-1500:]
    assert (target / "important.txt").read_text() == "x"
    d = DataDirectory(str(target))
    assert sorted(d.tefield_l0.get_list_of_events()) == [(1618, 1), (13790, 1)]
    assert sorted(d.tefield_l1.get_list_of_events()) == [(1618, 1), (13790, 1)]
    d.close()

    done = _run(tmp_path, "list.txt", "keep")                 # without -ow: already there
    assert done.returncode == 0
    assert "already in the target: 2" in done.stdout


def test_extract_refuses_unsafe_targets_and_reports_missing_events(tmp_path):
    (tmp_path / "list.txt").write_text("%s,1,13790\n" % SAMPLE)
    (tmp_path / "precious.txt").write_text("x")
    done = _run(tmp_path, "-ow", "list.txt", ".")
    assert done.returncode != 0 and "refusing" in done.stderr
    assert (tmp_path / "precious.txt").exists()

    (tmp_path / "missing.txt").write_text("%s,1,99\n" % SAMPLE)
    done = _run(tmp_path, "missing.txt", "out")
    assert done.returncode == 1 and "Not found" in done.stdout

    (tmp_path / "bad.txt").write_text("%s,1\n" % SAMPLE)
    done = _run(tmp_path, "bad.txt", "out2")
    assert done.returncode != 0 and "line 1" in done.stderr
    assert not (tmp_path / "out2").exists()
