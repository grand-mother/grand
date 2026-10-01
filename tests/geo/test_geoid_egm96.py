# -*- coding: utf-8 -*-
r"""The shipped EGM96 geoid map is the right way up (issue #250).

``data/egm96.png`` stored its rows south-first while TURTLE reads PNG rows
north-first, so every undulation was the value at the opposite latitude: the
North Pole read -29.5 m (the South Pole's value) and the GP300 site -7.75 m
instead of -61 m.  Its header also gave the far edges one grid step too far
(x1 = 360.25, y1 = 90.25 for a 0.25 degree grid of 1441 x 721 nodes), which
stretched the grid by up to a quarter of a degree.

The reference values are published EGM96 landmarks: the poles, and the global
minimum and maximum of the model, which sit on grid nodes.
"""

import numpy as np
import pytest

from grand.geo import topography
from grand.geo.coordinates import geoid_undulation

LANDMARKS = [
    ("North Pole", 90.0, 0.0, 13.61),
    ("South Pole", -90.0, 0.0, -29.53),
    ("global minimum, Indian Ocean", 4.75, 78.75, -106.99),
    ("global maximum, New Guinea", -8.25, 147.25, 85.39),
]


@pytest.mark.parametrize("name, latitude, longitude, expected", LANDMARKS,
                         ids=[landmark[0] for landmark in LANDMARKS])
def test_landmarks(name, latitude, longitude, expected):
    r"""Both entry points give the published EGM96 value, within 0.05 m."""
    assert float(geoid_undulation(latitude=latitude, longitude=longitude)) == pytest.approx(
        expected, abs=0.05), name
    assert float(topography.geoid_undulation(latitude=latitude, longitude=longitude)) == pytest.approx(
        expected, abs=0.05), name


def test_the_extremes_are_where_egm96_puts_them():
    r"""The minimum and maximum over the grid nodes are at 4.75N 78.75E and 8.25S 147.25E."""
    latitudes = np.arange(-89.75, 90.0, 0.25)
    longitudes = np.arange(0.0, 360.0, 0.25)
    lon, lat = np.meshgrid(longitudes, latitudes)
    values = np.asarray(geoid_undulation(latitude=lat.ravel(), longitude=lon.ravel())).reshape(lat.shape)
    low = np.unravel_index(np.argmin(values), values.shape)
    high = np.unravel_index(np.argmax(values), values.shape)
    assert (latitudes[low[0]], longitudes[low[1]]) == (4.75, 78.75)
    assert (latitudes[high[0]], longitudes[high[1]]) == (-8.25, 147.25)


def test_gp300_site():
    r"""At Dunhuang the geoid is about 61 m below the ellipsoid (it read -7.75 m before #250)."""
    assert float(geoid_undulation(latitude=40.98, longitude=93.95)) == pytest.approx(-61.04, abs=0.05)
