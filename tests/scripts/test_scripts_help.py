# -*- coding: utf-8 -*-
r"""#271: every script answers ``-h`` without error and without writing a file.

Ten of the fifteen scripts were exercised by no test, so an import broken by a
library change went unnoticed until someone ran the script.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = sorted(path.name for path in (ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_answers_help(script, tmp_path):
    argv = [sys.executable, str(ROOT / "scripts" / script)]
    if script != "get_version.py":        # it takes no options and only prints the version
        argv.append("-h")
    done = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg"))
    assert done.returncode == 0, done.stderr[-2000:]
    expected = "version=" if script == "get_version.py" else "usage:"
    assert expected in done.stdout
    assert list(tmp_path.iterdir()) == [], "%s -h wrote files" % script
