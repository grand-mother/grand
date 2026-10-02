# -*- coding: utf-8 -*-
r"""#218: the examples outside examples/old/ run, or say what they need."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CURRENT = [p for p in EXAMPLES.rglob("*") if "old" not in p.relative_to(EXAMPLES).parts]


def _run(*argv, cwd):
    return subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, capture_output=True, text=True,
                          timeout=900, env=dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg"))


@pytest.mark.parametrize("notebook", [p for p in CURRENT if p.suffix == ".ipynb"],
                         ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_every_notebook_is_valid_json(notebook):
    json.loads(notebook.read_text())          # one had merge-conflict markers


def test_no_stub_outside_old():
    for path in (p for p in CURRENT if p.suffix in (".py", ".ipynb")):
        text = path.read_text(errors="replace")
        assert "not working at the moment" not in text and "/home/grand/" not in text, path
        assert "test_efield.root" not in text, path


def test_the_index_lists_the_example_folders():
    index = (EXAMPLES / "README.md").read_text()
    for folder in ("aoi", "analysis", "dataio", "datalib", "eventviewer", "geo", "sim", "old"):
        assert "`%s/" % folder in index, folder


def test_shower_event_runs_on_the_committed_sample(tmp_path):
    done = _run(EXAMPLES / "sim" / "shower_event.py", cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]


def test_rf_chain_example_lists_its_choices(tmp_path):
    done = _run(EXAMPLES / "sim" / "rf_chain_example.py", "bogus", cwd=tmp_path)
    assert done.returncode == 2 and "choose from galactic" in done.stderr


def test_datafile_use_without_a_file_fails(tmp_path):
    done = _run(EXAMPLES / "dataio" / "datafile_use.py", cwd=tmp_path)
    assert done.returncode == 1 and "ROOT file" in done.stderr


def test_local_topography_downloads_only_when_asked(tmp_path):
    done = _run(EXAMPLES / "geo" / "local_topography.py", "-h", cwd=tmp_path)
    assert done.returncode == 0 and "--download" in done.stdout
