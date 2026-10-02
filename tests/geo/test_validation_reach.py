# -*- coding: utf-8 -*-
r"""#267: geo entry points the validation work had not reached.

Each refused input below was stored as given, or failed inside a C library
with an unrelated message.
"""

import numpy as np
import pytest

from grand import Geodetic, Geomagnet
from grand.geo import gull, turtle
from grand.geo.coordinates import (Horizontal, HorizontalRepresentation, HorizontalVector, LTP,
                                   SphericalRepresentation)

SITE = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)


@pytest.mark.parametrize("make, match", [
    (lambda: SphericalRepresentation(theta=200.0, phi=0.0, r=1.0), "theta"),
    (lambda: SphericalRepresentation(theta=10.0, phi=0.0, r=-1.0), "'r'"),
    (lambda: HorizontalRepresentation(azimuth=0.0, elevation=95.0, norm=1.0), "elevation"),
    (lambda: Geodetic(latitude=1.0, longitude=400.0, height=0.0), "longitude"),
    (lambda: Geodetic(latitude=1.0, longitude=4.0, height=-1e7), "height"),
    (lambda: LTP(x=1.0, y=0.0, z=0.0, location=SITE, orientation="XYZ"), "one from each of E/W"),
    (lambda: LTP(x=1.0, y=0.0, z=0.0, location=SITE, orientation="ENN"), "one from each of E/W"),
    (lambda: LTP(x=1.0, y=0.0, z=0.0, location=SITE, orientation="ENU", rotation=np.ones((3, 3))),
     "rotation"),
    (lambda: Geomagnet(model="XXX", location=SITE), "available: IGRF13"),
])
def test_invalid_values_are_refused(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_valid_values_still_work():
    SphericalRepresentation(theta=180.0, phi=0.0, r=0.0)
    LTP(x=1.0, y=0.0, z=0.0, location=SITE, orientation="nwu", rotation=np.eye(3))
    assert np.ravel(Geodetic(latitude=1.0, longitude=-360.0, height=0.0).longitude)[0] == pytest.approx(0.0)


def test_horizontal_vector_takes_what_horizontal_takes():
    vector = HorizontalVector(azimuth=0.0, elevation=10.0, norm=1.0, location=SITE)
    np.testing.assert_allclose(np.asarray(vector), np.asarray(Horizontal(azimuth=0.0, elevation=10.0,
                                                                        norm=1.0, location=SITE)))


@pytest.mark.parametrize("value", ["45", 1j])
def test_turtle_refuses_what_is_not_a_real_number(value):
    with pytest.raises(TypeError, match="real number"):
        turtle.ecef_from_geodetic(value, 3.0, 0.0)
    assert np.isnan(turtle.ecef_from_geodetic(None, 3.0, 0.0)).all()      # None stays NaN


def test_turtle_and_gull_name_missing_files():
    with pytest.raises(FileNotFoundError, match="no map file"):
        turtle.Map("/nonexistent/map.png")
    with pytest.raises(FileNotFoundError, match="no geomagnetic model file"):
        gull.Snapshot("/nonexistent/model.COF")


def test_gull_snapshot_has_a_working_default_and_takes_a_path(tmp_path):
    from pathlib import Path

    from grand.geo.geomagnet import DATADIR

    assert gull.Snapshot() is not None
    assert gull.Snapshot(Path(DATADIR) / "WMM2020.COF") is not None


def test_map_elevation_keeps_the_input_shape():
    from grand.geo.coordinates import DATADIR

    egm = turtle.Map(str(DATADIR) + "/egm96.png")
    assert egm.elevation(np.full((2, 3), 93.95), np.full((2, 3), 40.98)).shape == (2, 3)
