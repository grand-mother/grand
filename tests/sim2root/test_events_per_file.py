# -*- coding: utf-8 -*-
r"""#223: ``sim2root.py -ef N`` and inputs from different runs.

``-ef 1`` did not split, ``-ef N`` left a trailing empty file set whenever N
divided the number of events, and inputs from different runs were merged
under the first run's number.
"""

import os
import subprocess
import sys

import pytest

from tests.sim2root.test_sim2root_beta_fixes import RUN_13790, SIM2ROOT, ZHAIRES, _convert

pytestmark = pytest.mark.skipif(not (ZHAIRES / RUN_13790).is_dir(), reason="the committed samples are not present")


def _sim2root(tmp_path, raw, *args):
    out = tmp_path / ("out" + "".join(args).replace("-", "_"))
    out.mkdir()
    done = subprocess.run([sys.executable, str(SIM2ROOT), str(raw), *args, "-sl", "GP300", "-o", str(out)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=900,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    return done, out


def _entries_per_efield_file(out):
    from grand.dataio import TEfield

    counts = []
    for path in sorted(out.rglob("efield_*.root")):
        with TEfield(str(path)) as tree:
            counts.append(tree.get_number_of_entries())
    return counts


@pytest.mark.parametrize("per_file, expected", [("1", [1, 1, 1, 1]), ("2", [2, 2]), ("3", [3, 1]),
                                                ("4", [4])])
def test_events_per_file(tmp_path, per_file, expected):
    raw = _convert(tmp_path, (5, 6, 7, 8), "four.rawroot")
    done, out = _sim2root(tmp_path, raw, "-ef", per_file)
    assert done.returncode == 0, done.stderr[-2000:]
    assert _entries_per_efield_file(out) == expected
    for kind in ("shower", "showersim"):
        assert len(list(out.rglob(kind + "_*.root"))) == len(expected)
    assert not list(out.rglob("efield.root")), "a placeholder file was left"


def test_different_runs_are_refused_without_a_run_number(tmp_path):
    first = _convert(tmp_path, (5,), "run1.rawroot")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, str(ZHAIRES / "ZHAireSRawToRawROOT.py"), "./" + RUN_13790, "standard",
                    "13", "6", str(tmp_path / "run13.rawroot")],
                   cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900, check=True)
    folder = tmp_path / "raw"
    folder.mkdir()
    first.rename(folder / first.name)
    (tmp_path / "run13.rawroot").rename(folder / "run13.rawroot")

    done, out = _sim2root(tmp_path, folder)
    assert done.returncode != 0
    assert "runs 1 and 13" in done.stderr and "-ru" in done.stderr
    assert not list(out.rglob("*.root")), "files were written before the refusal"

    done, out = _sim2root(tmp_path, folder, "-ru", "7")
    assert done.returncode == 0, done.stderr[-2000:]
