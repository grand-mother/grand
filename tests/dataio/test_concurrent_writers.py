# -*- coding: utf-8 -*-
r"""#281: several processes writing one ROOT file.

They used to interleave their writes: events were lost while every process
reported success, or the file was left unreadable.  Now one process writes at a
time, and the others are refused with a clear message; nothing reported as
written is lost and the file stays readable.
"""

import pathlib
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

pytest.importorskip("fcntl")

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

WRITER = textwrap.dedent('''
    import sys, time
    import numpy as np
    from grand.dataio import TVoltage
    out, wid, n, hold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    ok = 0
    try:
        for i in range(n):
            t = TVoltage(out)
            t.run_number = 1; t.event_number = wid * 1000 + i; t.du_count = 1; t.du_id = [1]
            t.trace_ch = np.ones((1, 3, 256), np.float32).tolist()
            t.fill(); t.write()
            time.sleep(hold)            # still holding the file
            t.stop_using(); ok += 1
    finally:
        print("ok=%d" % ok, flush=True)
''')


def _start(tmp_path, out, wid, n, hold=0.0):
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    return subprocess.Popen([sys.executable, str(script), str(out), str(wid), str(n), str(hold)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path)


def _finish(proc):
    out, err = proc.communicate(timeout=300)
    return int(out.strip().rsplit("ok=", 1)[1]), err


def _entries(path):
    import ROOT

    f = ROOT.TFile.Open(str(path))
    assert f and not f.IsZombie()
    n = int(f.Get("tvoltage").GetEntries())
    f.Close()
    return n


def test_a_file_another_process_is_writing_is_refused(tmp_path):
    out = tmp_path / "out.root"
    holder = _start(tmp_path, out, 1, 1, hold=8.0)
    deadline = time.time() + 120
    while not out.exists() and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(1.0)
    second = _start(tmp_path, out, 2, 1)
    ok2, err2 = _finish(second)
    ok1, _ = _finish(holder)
    assert (ok1, ok2) == (1, 0)
    assert "another process is writing it" in err2 or "being written by another process" in err2
    assert _entries(out) == 1


def test_a_write_after_another_process_wrote_is_refused(tmp_path):
    from grand.dataio import TVoltage

    out = tmp_path / "out.root"
    assert _finish(_start(tmp_path, out, 9, 1))[0] == 1
    stale = TVoltage(str(out))                         # opened here, then written elsewhere
    assert _finish(_start(tmp_path, out, 1, 1))[0] == 1
    stale.run_number = 1
    stale.event_number = 5
    stale.du_count = 1
    stale.du_id = [1]
    stale.trace_ch = np.ones((1, 3, 256), np.float32).tolist()
    with pytest.raises(OSError, match="changed by another process after this one opened it"):
        stale.fill()
    stale.stop_using()
    assert _entries(out) == 2                          # intact, nothing lost


@pytest.mark.parametrize("existing", [True, False], ids=["existing_file", "new_file"])
def test_concurrent_writers_lose_nothing_they_report(tmp_path, existing):
    out = tmp_path / "out.root"
    if existing:
        assert _finish(_start(tmp_path, out, 9, 1))[0] == 1
    procs = [_start(tmp_path, out, w, 10) for w in (1, 2, 3)]
    reported = sum(_finish(p)[0] for p in procs)
    assert reported >= 1
    assert _entries(out) == reported + (1 if existing else 0)


def test_one_process_still_writes_a_file_repeatedly(tmp_path):
    out = tmp_path / "out.root"
    assert _finish(_start(tmp_path, out, 1, 5))[0] == 5
    assert _finish(_start(tmp_path, out, 2, 5))[0] == 5
    assert _entries(out) == 10
