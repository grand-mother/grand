# -*- coding: utf-8 -*-
r"""#184: the open_grand_* scripts pasted the file name into Python code.

A quote in the name broke the session, and a crafted name ran arbitrary code.
The command they build is checked here, without starting a shell: ``execlp``
is replaced by a stub that records the command and runs it with ``-c``, minus
the interactive part.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Pasted into the old command, this called open() while building the name
NASTY = "x' + str(open('PWNED', 'w').write('1')) + '"

STUB = r"""
import os, runpy, sys
def fake(interp, *argv):
    code = argv[-1]
    # Run the whole command; it must fail as a missing file, never by
    # running the injected code
    try:
        exec(code, {})
    except SyntaxError:
        print("SYNTAXERROR")
    except Exception as error:
        print("OPENERROR", type(error).__name__)
    sys.exit(0)
os.execlp = fake
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.mark.parametrize("script", ["open_grand_file.py", "open_grand_directory.py",
                                    "open_grand_analysis_prompt.py"])
def test_the_name_is_never_code(tmp_path, script):
    done = subprocess.run([sys.executable, "-c", STUB, str(ROOT / "scripts" / script), "-s",
                           NASTY], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert not (tmp_path / "PWNED").exists(), done.stdout + done.stderr
    assert "SYNTAXERROR" not in done.stdout
    assert "OPENERROR" in done.stdout, done.stdout + done.stderr[-1500:]
