# -*- coding: utf-8 -*-
r"""#224: sim2root and the ZHAireS converter fail with a non-zero status and
leave no junk behind.

A missing input was created empty and the run died on an unbound variable;
no input, a missing ``-sl`` and a failed rename exited 0; a failure left the
placeholder files (``run.root``, ``efield.root``, ...) in the output folder;
``-fo`` on a used folder added a second file set that broke the later steps;
and the ZHAireS converter had no argparse and exited 0 on its errors.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tests.sim2root.test_sim2root_beta_fixes import RUN_13790, SIM2ROOT, ZHAIRES, _convert

pytestmark = pytest.mark.skipif(not (ZHAIRES / RUN_13790).is_dir(), reason="the committed samples are not present")
CONVERTER = ZHAIRES / "ZHAireSRawToRawROOT.py"


def _env():
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _sim2root(tmp_path, *args):
    return subprocess.run([sys.executable, str(SIM2ROOT), *args], cwd=tmp_path, capture_output=True,
                          text=True, timeout=900, env=_env())


def test_a_missing_input_is_refused_and_not_created(tmp_path):
    missing = tmp_path / "nofile.rawroot"
    out = tmp_path / "out"
    done = _sim2root(tmp_path, str(missing), "-sl", "GP300", "-o", str(out))
    assert done.returncode == 1
    assert "no such input file" in done.stderr
    assert not missing.exists()
    assert not out.exists() or not any(out.iterdir())


def test_no_input_and_no_layout_exit_non_zero(tmp_path):
    (tmp_path / "empty").mkdir()
    done = _sim2root(tmp_path, str(tmp_path / "empty"), "-sl", "GP300")
    assert done.returncode == 1 and "no .rawroot files" in done.stderr
    raw = _convert(tmp_path, (5,), "one.rawroot")
    done = _sim2root(tmp_path, str(raw))
    assert done.returncode == 1 and "-sl" in done.stderr


def test_a_used_forced_folder_needs_overwrite(tmp_path):
    raw = _convert(tmp_path, (5,), "one.rawroot")
    used = tmp_path / "used"
    used.mkdir()
    (used / "earlier.root").write_text("")
    done = _sim2root(tmp_path, str(raw), "-sl", "GP300", "-fo", str(used))
    assert done.returncode == 1 and "--overwrite" in done.stderr
    assert sorted(p.name for p in used.iterdir()) == ["earlier.root"]
    done = _sim2root(tmp_path, str(raw), "-sl", "GP300", "-fo", str(used), "--overwrite")
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_failure_removes_the_partial_output(tmp_path):
    raw = _convert(tmp_path, (5, 6), "two.rawroot")
    out = tmp_path / "out"
    out.mkdir()
    # Fails on the second event, after the output folder and files exist
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        sys.argv = ["sim2root.py", %r, "-sl", "GP300", "-o", %r]
        import sim2root
        original, calls = sim2root.rawshower2grandroot, []
        def failing(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected failure")
            return original(*args)
        sim2root.rawshower2grandroot = failing
        sim2root.main()
    """ % (str(SIM2ROOT.parent), str(raw), str(out)))
    done = subprocess.run([sys.executable, "-c", code], cwd=SIM2ROOT.parent, capture_output=True,
                          text=True, timeout=900, env=_env())
    assert done.returncode != 0 and "injected failure" in done.stderr
    assert list(out.iterdir()) == [], "the failed run left %s" % list(out.rglob("*"))


def test_the_zhaires_converter_has_a_command_line(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(CONVERTER), *args], cwd=tmp_path, capture_output=True,
                              text=True, timeout=300, env=_env())

    done = run("--help")
    assert done.returncode == 0 and "usage:" in done.stdout and not (tmp_path / "--help").exists()
    done = run(str(tmp_path / "nowhere"))
    assert done.returncode == 1 and "no such folder" in done.stderr
    done = run(str(tmp_path), "full", "1", "1", "x.rawroot")
    assert done.returncode == 2 and "standard" in done.stderr
    done = run(str(tmp_path), "standard", "1")
    assert done.returncode == 2
    done = run(str(tmp_path), "standard", "1", "1", str(tmp_path / "x.rawroot"))   # no .sry
    assert done.returncode == 1 and "failed" in done.stderr
    assert not (tmp_path / "x.rawroot").exists()


@pytest.mark.parametrize("function", ["GetThinningRelativeEnergyFromSry", "GetPrimaryFromSry",
                                      "GetTaskNameFromSry"])
def test_the_aires_readers_raise_instead_of_exiting(tmp_path, function):
    # exit() in these ended whatever program had imported them (#224)
    import sim2root.ZHAireSRawRoot.AiresInfoFunctionsGRANDROOT as aires

    sry = tmp_path / "empty.sry"
    sry.write_text("nothing useful\n")
    with pytest.raises(ValueError, match="GRANDlib: %s: no " % function):
        getattr(aires, function)(str(sry))
