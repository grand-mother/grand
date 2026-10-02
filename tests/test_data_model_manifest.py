# -*- coding: utf-8 -*-
r"""#279: the data model is checked before use.

A missing file failed with whatever library read it; a damaged one that
still parsed changed the voltages silently; the downloader said "up to date"
with noise/ or topography/ deleted; and convert_voltage2adc.py needed psutil
even for -h.
"""

import os
import pathlib
import subprocess
import sys

import pytest

from grand.basis import data_model

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _install(root):
    for directory, name, text in (("detector", "a.s2p", "1 2 3\n4 5 6\n"), ("noise", "b.npy", "x"),
                                  ("topography", "c.hgt", "y")):
        (root / directory).mkdir(parents=True)
        (root / directory / name).write_text(text)


def test_the_manifest_records_and_verifies_an_installation(tmp_path):
    _install(tmp_path)
    assert data_model.write_manifest(str(tmp_path), "v1") == 3
    assert data_model.verify(str(tmp_path), sums=True) == []
    (tmp_path / "detector" / "a.s2p").write_text("1 2 3\n")           # cut in half
    (tmp_path / "noise" / "b.npy").unlink()
    problems = data_model.verify(str(tmp_path))
    assert any("a.s2p has" in p for p in problems) and any("b.npy is missing" in p for p in problems)


def test_a_missing_directory_is_reported_without_a_manifest(tmp_path):
    _install(tmp_path)
    (tmp_path / "noise" / "b.npy").unlink()
    (tmp_path / "noise").rmdir()
    assert data_model.verify(str(tmp_path)) == ["noise/ is missing"]


def test_check_names_the_remedy(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="data model incomplete.*download_data_grand"):
        data_model.check(tmp_path / "nothing.s2p", "test")
    monkeypatch.setattr(data_model, "GRAND_DATA_PATH", str(tmp_path))
    _install(tmp_path)
    monkeypatch.setattr(data_model, "_manifest", {os.path.join("detector", "a.s2p"): {"size": 1, "sha256": ""}})
    with pytest.raises(ValueError, match="data model damaged"):
        data_model.check(tmp_path / "detector" / "a.s2p", "test")


@pytest.mark.skipif(not (ROOT / "data" / "detector").exists(), reason="needs the data model")
def test_a_damaged_rf_chain_file_is_refused(monkeypatch):
    from grand import GRAND_DATA_PATH
    from grand.sim.detector.rf_chain import RFChain

    files = {}
    for base, _, names in os.walk(os.path.join(GRAND_DATA_PATH, "detector", "RFchain_v2")):
        for name in names:
            path = os.path.join(base, name)
            files[os.path.relpath(path, GRAND_DATA_PATH)] = {"size": os.path.getsize(path) + 1, "sha256": ""}
    monkeypatch.setattr(data_model, "_manifest", files)
    with pytest.raises(ValueError, match="data model damaged"):
        RFChain()


def test_voltage2adc_help_needs_no_psutil():
    code = ("import sys, runpy; sys.modules['psutil'] = None; "
            "sys.argv = ['convert_voltage2adc.py', '-h']; "
            "runpy.run_path(%r, run_name='__main__')" % str(ROOT / "scripts" / "convert_voltage2adc.py"))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert done.returncode == 0, done.stderr[-1500:]
    assert "usage:" in done.stdout

