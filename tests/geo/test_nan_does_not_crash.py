# -*- coding: utf-8 -*-
r"""A NaN position must not crash the interpreter (issue #262).

libturtle's elevation lookups segfaulted on a NaN latitude or longitude, killing
the whole Python process with no traceback.  Each call below runs in its own
subprocess, so a crash fails the test instead of taking pytest down with it.
A NaN point now gives a NaN elevation with a ``GRANDlibWarning``; ``None`` is
refused with a ``TypeError``.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HAVE_TILE = (ROOT / "data" / "topography" / "N41E096.hgt").exists()

SETUP = """
import pathlib
import warnings
import numpy as np
from grand.basis.validate import GRANDlibWarning
from grand.geo import topography, turtle
from grand.geo.coordinates import Geodetic, GRANDCS, geoid_undulation
egm96 = str(topography._get_geoid().path)
nan = float('nan')
"""

NAN_CALLS = {
    "geoid_undulation": "geoid_undulation(nan, 3.)",
    "topography_geoid_undulation": "topography.geoid_undulation(latitude=nan, longitude=3.)",
    "Map.elevation": "turtle.Map(egm96).elevation(3., nan)",
    "Map.elevation_array": "turtle.Map(egm96).elevation(np.array([3., nan]), np.array([45., 45.]))",
}
TILE_CALLS = {
    "Stack.elevation": "turtle.Stack(str(pathlib.Path(egm96).parent / 'topography')).elevation(nan, 96.5)",
    "elevation_geoid": "topography.elevation(Geodetic(latitude=nan, longitude=96.5, height=0.))",
    "elevation_ellipsoid": "topography.elevation(Geodetic(latitude=nan, longitude=96.5, height=0.),"
                           " reference='ELLIPSOID')",
    "elevation_local": "topography.elevation(GRANDCS(x=np.array([0., nan]), y=np.zeros(2), z=np.zeros(2),"
                       " location=Geodetic(latitude=41.5, longitude=96.5, height=0.)), reference='LOCAL')",
}


def _run(code):
    return subprocess.run([sys.executable, "-c", SETUP + code], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)


def _check_nan_and_warning(call):
    done = _run("""
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    value = np.atleast_1d(%s)
assert np.isnan(value[-1]), value
assert any(issubclass(w.category, GRANDlibWarning) and 'non-finite' in str(w.message)
           for w in caught), [str(w.message) for w in caught]
print('OK')
""" % call)
    assert done.returncode == 0, "exit code %d (a negative code is a crash): %s" % (
        done.returncode, done.stderr[-1500:])
    assert done.stdout.strip().endswith("OK")


@pytest.mark.parametrize("call", NAN_CALLS.values(), ids=NAN_CALLS.keys())
def test_a_nan_gives_nan_and_a_warning(call):
    _check_nan_and_warning(call)


@pytest.mark.skipif(not HAVE_TILE, reason="the N41E096 elevation tile is not in data/topography")
@pytest.mark.parametrize("call", TILE_CALLS.values(), ids=TILE_CALLS.keys())
def test_a_nan_gives_nan_and_a_warning_with_tiles(call):
    _check_nan_and_warning(call)


def test_none_is_refused():
    done = _run("""
try:
    turtle.Map(egm96).elevation(None, 45.)
except TypeError as error:
    assert 'GRANDlib: Map.elevation' in str(error), error
    print('OK')
""")
    assert done.returncode == 0, done.stderr[-1500:]
    assert done.stdout.strip().endswith("OK")


def test_finite_points_are_unchanged():
    from grand.geo.coordinates import geoid_undulation

    assert float(geoid_undulation(latitude=40.98, longitude=93.95)) == pytest.approx(-61.0414, abs=1e-3)
