# -*- coding: utf-8 -*-
r"""#280: environment problems name their remedy.

A missing compiled core, a missing optional package, a point outside the
downloaded topography and a read-only output folder each failed with a bare
error, or silently, far from the cause.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from grand.basis.validate import GRANDlibWarning

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def _import_without(blocked, module):
    """Imports `module` in a fresh interpreter where `blocked` cannot be imported."""
    code = textwrap.dedent("""
        import sys
        sys.modules[%r] = None
        try:
            import %s
        except ImportError as error:
            print(error)
        else:
            print("imported")
    """ % (blocked, module))
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    return run.stdout + run.stderr


@pytest.mark.parametrize("module", ["grand.geo.topography", "grand.geo.turtle", "grand.geo.gull"])
def test_a_missing_core_says_how_to_build_it(module):
    out = _import_without("grand._core", module)
    assert "compiled core" in out and "env/setup.sh" in out


def test_a_missing_iminuit_names_the_extra():
    out = _import_without("iminuit", "grand.analysis.fitting.adf")
    assert 'pip install -e ".[analysis]"' in out


def test_a_point_outside_the_tiles_warns():
    from grand import Geodetic, topography

    # Far from any site: no tile is ever downloaded there.
    with pytest.warns(GRANDlibWarning, match="1 of 1 points are outside the loaded topography tiles"):
        topography.elevation(Geodetic(latitude=-60.0, longitude=10.0, height=0.0))


@pytest.mark.skipif(not (ROOT / "data" / "detector").exists(), reason="needs the data model")
def test_a_read_only_output_folder_is_refused_before_computing(tmp_path, monkeypatch):
    from grand import Efield2Voltage

    real_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode: False if pathlib.Path(path) == tmp_path
                        else real_access(path, mode))
    with pytest.raises(PermissionError, match="cannot write in the output folder"):
        Efield2Voltage(str(SAMPLE), "out.root", output_directory=str(tmp_path), seed=1)
