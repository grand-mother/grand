# -*- coding: utf-8 -*-
r"""Names found on disk reach external programs as arguments, not as shell code.

The ZHAireS helpers and the simulation-pipeline scripts used to build shell
command strings from simulation paths, task names and directory names. A name
containing ``$( )``, backticks or ``;`` was then executed. Here such names
must arrive verbatim at the program, and nothing else may run.
"""

import json
import os
import pathlib
import stat
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZHAIRES = ROOT / "sim2root" / "ZHAireSRawRoot"
COMMON = ROOT / "sim2root" / "Common"

NASTY = "t$(touch PWNED);x"


def _fake_program(path, log):
    r"""An executable that records its arguments as JSON lines in ``log``."""
    path.write_text("#!%s\nimport json, sys\nwith open(%r, 'a') as f:\n"
                    "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n" % (sys.executable, str(log)))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_aires_export_gets_the_task_name_as_one_argument(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / (NASTY + ".idf")).write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.jsonl"
    _fake_program(bindir / "AiresExport", log)
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "import AiresInfoFunctionsGRANDROOT as A\n"
            "A.GetLongitudinalTable(%r, 1205)\n" % (str(ROOT), str(ZHAIRES), str(sim)))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                   text=True, timeout=120, env=dict(os.environ, AIRESBINDIR=str(bindir)))
    assert not (tmp_path / "PWNED").exists() and not (sim / "PWNED").exists()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls, "AiresExport was not called"
    assert any(arg.endswith(NASTY) for arg in calls[0])


def test_pipeline_steps_pass_arguments_without_a_shell(tmp_path):
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "fake_python"
    _fake_program(fake, log)
    # INPUTDIR and EXTRA as a user, or a directory found on disk, could name them
    done = subprocess.run([sys.executable, str(COMMON / "RunSimPipe.py"), "sim; touch PWNED #",
                           "e$(touch PWNED2)"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONINTERPRETER=str(fake)))
    assert not (tmp_path / "PWNED").exists() and not (tmp_path / "PWNED2").exists(), done.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls, done.stdout[-1500:] + done.stderr[-1500:]
    assert "sim; touch PWNED #" in calls[0]
    assert "e$(touch PWNED2)" in calls[0]
