# -*- coding: utf-8 -*-
r"""The simulation-pipeline scripts show each step's output as it comes (#121).

``RunSimPipe*.py`` captured it: all of a step's output at once when it
ended, stderr as a raw byte string, or, in the streaming variant, through a
pipe that was reported to freeze on long runs.  Each step now writes
directly to the scripts' own output.
"""

import importlib.util
import pathlib
import sys

import pytest

COMMON = pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
SCRIPTS = ["RunSimPipe.py", "RunSimPipeNoJitter.py", "RunSimPipeADCNoise.py"]


def _pipeline_step():
    spec = importlib.util.spec_from_file_location("pipeline_step", COMMON / "pipeline_step.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_step_with_a_lot_of_output_finishes_and_keeps_it_in_order(capfd):
    r"""200,000 lines on stdout and stderr: nothing to fill up, nothing lost."""
    step = _pipeline_step()
    cmd = ('%s -c "import sys; [(print(i), print(-i, file=sys.stderr))'
           ' for i in range(100000)]"' % sys.executable)
    assert step.run_step(cmd) == 0

    out, err = capfd.readouterr()
    lines = out.splitlines()
    assert lines[0].startswith("about to run: ")
    assert lines[1:] == [str(i) for i in range(100000)]
    assert err.splitlines() == [str(-i) for i in range(100000)]


def test_a_failed_step_is_reported_and_the_pipeline_goes_on(capfd):
    r"""The status is returned and printed; nothing raises, as before."""
    step = _pipeline_step()
    assert step.run_step("%s -c 'import sys; sys.exit(3)'" % sys.executable) == 3
    assert "WARNING: step exited with status 3" in capfd.readouterr().err


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_scripts_run_every_step_through_it(script):
    r"""No step is captured any more; the commands themselves are unchanged."""
    source = (COMMON / script).read_text()
    assert "from pipeline_step import run_step" in source
    assert "communicate(" not in source and "Popen(" not in source
    assert source.count("run_step(cmd)") == 4
    compile(source, script, "exec")
