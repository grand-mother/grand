# -*- coding: utf-8 -*-
r"""The commands the documentation gives, run as written (#257).

``sim2root.rst``, ``simulation.rst`` and ``quickstart.rst`` gave commands that
failed: the CoREAS converter without ``-d``, ``sim2root.py`` without ``-sl``,
and the voltage and ADC steps on a single file.  These tests read the
commands out of the pages, fill in the placeholders with the committed
samples, and run them, so that the pages cannot drift from the scripts again.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
ZHAIRES = sorted((ROOT / "sim2root" / "ZHAireSRawRoot").glob("*.rawroot"))
COREAS = ROOT / "sim2root" / "CoREASRawRoot"

pytestmark = pytest.mark.skipif(len(ZHAIRES) != 2 or not (COREAS / "proton").is_dir(),
                                reason="the committed samples are not present")


def _blocks(page, language):
    r"""Returns the code blocks of `language` in `page`, as lists of lines."""
    lines = (DOCS / page).read_text().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip() == ".. code-block:: %s" % language:
            i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                body.append(lines[i][4:])
                i += 1
            blocks.append([b for b in body if b.strip()])
        else:
            i += 1
    return blocks


def _commands(page):
    r"""Returns the ``python`` command lines of the bash blocks of `page`."""
    out = []
    for block in _blocks(page, "bash"):
        joined = "\n".join(block).replace("\\\n", " ")
        out += [line.split() for line in joined.splitlines()
                if line.startswith(("python ", "python3 "))]
    return out


def _run(argv, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable] + argv[1:], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, "%s\n%s" % (" ".join(argv), done.stderr[-2000:])


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    r"""Runs ``sim2root.py`` as ``sim2root.rst`` shows; returns the folder."""
    work = tmp_path_factory.mktemp("docs")
    for raw in ZHAIRES:
        shutil.copy(raw, work)
    (sim2root,) = [c for c in _commands("sim2root.rst") if c[1].endswith("sim2root.py")]
    argv = [a.replace("sim2root/Common/sim2root.py", str(ROOT / "sim2root" / "Common" / "sim2root.py"))
            for a in sim2root]
    assert "<path>/*.rawroot" in argv and "-sl" in argv
    i = argv.index("<path>/*.rawroot")
    argv[i:i + 1] = [r.name for r in ZHAIRES]
    _run(argv, work)
    (folder,) = work.glob("sim_*")
    return folder


def test_the_coreas_converter_command(tmp_path):
    (command,) = [c for c in _commands("sim2root.rst") if c[1] == "CoreasToRawROOT.py"]
    # As a reader runs it: in the folder, next to the committed sample
    shutil.copytree(COREAS, tmp_path / "CoREASRawRoot")
    _run(command, tmp_path / "CoREASRawRoot")
    assert list((tmp_path / "CoREASRawRoot" / "converted").glob("*.rawroot"))


def test_the_voltage_and_adc_commands(simulation):
    commands = [c for c in _commands("simulation.rst")
                if c[1].startswith("scripts/convert_")]
    assert [c[1] for c in commands] == ["scripts/convert_efield2voltage.py",
                                        "scripts/convert_voltage2adc.py"]
    for command in commands:
        argv = [str(ROOT / a) if a.startswith("scripts/") else a for a in command]
        argv = [str(simulation) if a == "my_simulation" else a for a in argv]
        _run(argv, simulation.parent)
    assert list(simulation.glob("voltage_*_L0_*.root"))
    assert list(simulation.glob("adc_*_L1_*.root"))


@pytest.mark.parametrize("page", ["quickstart.rst", "simulation.rst"])
def test_the_python_example(simulation, page, tmp_path):
    (block,) = [b for b in _blocks(page, "python") if any("Efield2Voltage(" in line for line in b)][:1]
    code = "\n".join(block).replace('"my_simulation"', repr(str(simulation)))
    code = re.sub(r"output_directory=\"\.\"", "output_directory=%r" % str(tmp_path), code)
    script = tmp_path / "example.py"
    script.write_text(code + "\n")
    _run([sys.executable, str(script)], tmp_path)
    assert (tmp_path / "voltage.root").is_file()


def test_voltage2adc_says_what_it_needs(tmp_path):
    r"""Given a folder without a voltage file, it failed with ``IndexError``."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_voltage2adc.py"), str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "GRANDlib: convert_voltage2adc: no voltage_*_L<level>_*.root" in done.stderr
