# -*- coding: utf-8 -*-
r"""The code on the cheat sheet runs as written.

The cheat sheet is the page people copy from most, and its code is not
executed when the documentation is built.  Its Python and shell blocks are run
here on a copy of the committed simulation folder, named ``my_simulation`` as
on the page.
"""

import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "source" / "cheatsheet.rst"
SAMPLE = REPO / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="the sample run is not present")


def _blocks(language):
    """The code blocks of the page in `language`, dedented."""
    text = PAGE.read_text(encoding="utf-8")
    blocks = []
    for match in re.finditer(r"^\.\. code-block:: %s\n\n((?:    .*\n|\n)+)" % language, text, re.M):
        lines = match.group(1).rstrip("\n").split("\n")
        blocks.append("\n".join(line[4:] for line in lines))
    return blocks


@pytest.fixture
def folder(tmp_path, monkeypatch):
    shutil.copytree(SAMPLE, tmp_path / "my_simulation")
    # The page's simulation folder holds only what sim2root.py and the
    # e-field step write; the shell block writes the voltage and ADC files
    for stale in list((tmp_path / "my_simulation").glob("voltage_*")) + \
                 list((tmp_path / "my_simulation").glob("adc_*")):
        stale.unlink()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_python_block_runs(folder):
    (code,) = _blocks("python")
    exec(compile(code, str(PAGE), "exec"), {"__name__": "cheatsheet"})
    assert (folder / "out" / "voltage.root").is_file()


def test_the_shell_block_runs(folder):
    (code,) = _blocks("bash")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    for line in code.splitlines():
        args = shlex.split(line)
        assert args[0] == "python" and args[1].startswith("scripts/")
        subprocess.run([sys.executable, str(REPO / args[1])] + args[2:], cwd=folder, env=env,
                       check=True, capture_output=True)
    names = sorted(p.name for p in (folder / "my_simulation").glob("*")
                   if p.name.startswith(("voltage_", "adc_")))
    assert names == ["adc_1618-13790_L1_0000.root", "voltage_1618-13790_L0_0000.root"]
