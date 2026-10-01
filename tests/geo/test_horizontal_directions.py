# -*- coding: utf-8 -*-
r"""`Horizontal` from local (east, north, up) positions (#270).

``test_coordinates.py::test_horizontal`` has its body commented out, so
swapping the azimuth's ``arctan2`` arguments passed every test.
"""

import numpy as np
import pytest

from grand.geo.coordinates import ECEF, LTP, Geodetic, Horizontal

SITE = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)


@pytest.mark.parametrize("east, north, up, azimuth, elevation", [
    (1, 0, 0, 90.0, 0.0),
    (0, 1, 0, 0.0, 0.0),
    (1, 1, 0, 45.0, 0.0),
    (-1, 0, 0, -90.0, 0.0),
    (0, 0, 1, None, 90.0),
])
def test_azimuth_from_north_towards_east(east, north, up, azimuth, elevation):
    point = LTP(x=east, y=north, z=up, location=SITE, orientation="ENU", magnetic=False)
    h = Horizontal(point, location=SITE)
    if azimuth is not None:
        difference = (float(np.ravel(h.azimuth)[0]) - azimuth + 180) % 360 - 180
        assert difference == pytest.approx(0, abs=1e-6)
    assert float(np.ravel(h.elevation)[0]) == pytest.approx(elevation, abs=1e-6)


def test_round_trip_through_ecef():
    point = LTP(x=3.0, y=-4.0, z=2.0, location=SITE, orientation="ENU", magnetic=False)
    back = Horizontal(point, location=SITE).horizontal_to_ecef()
    assert np.allclose(np.ravel(back), np.ravel(ECEF(point)), atol=1e-6)
