# -*- coding: utf-8 -*-
r"""The "Read one event" recipes of ``datamodel.rst``, run as written (#192).

They are ``code-block``s, not executed when the docs are built, because they
read the committed sample; these tests run them and check what they print.
"""

import os
import pathlib
import subprocess
import sys

import pytest

from tests.test_documented_commands import _blocks

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def _recipes():
    blocks = _blocks("datamodel.rst", "python")
    return [b for b in blocks if any("sim_Xiaodushan" in line for line in b)]


def test_the_page_has_both_recipes():
    recipes = ["\n".join(b) for b in _recipes()]
    assert len(recipes) == 2
    assert "grand.dataio" in recipes[0] and "grand.aoi" in recipes[1]


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="the committed sample is not present")
@pytest.mark.parametrize("index", [0, 1], ids=["dataio", "aoi"])
def test_a_recipe_prints_the_first_event(index):
    code = "\n".join(_recipes()[index])
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "run 1, event 13790" in done.stdout
    assert "zenith 79.43 deg, azimuth 310.00 deg" in done.stdout
    assert "energy 3.87e+09 GeV, Xmax 794.6 g/cm2" in done.stdout
    assert "antennas " in done.stdout
