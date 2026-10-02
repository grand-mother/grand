# -*- coding: utf-8 -*-
r"""#261: what the geo docstrings now state, checked.

``Geomagnet`` did not give its unit or frame; ``Geodetic`` claimed a height of
zero is sea level and a longitude range it does not enforce; the two
``geoid_undulation`` functions had different signatures, so
``grand.geoid_undulation(40.98, 93.95)`` raised ``TypeError``.
"""

import numpy as np
import pytest

import grand
from grand import ECEF, GRANDCS, Geodetic, Geomagnet
from grand.geo import coordinates, topography

SITE = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)


def test_the_field_is_tesla_east_north_up_whatever_the_input_frame():
    field = np.ravel(Geomagnet(location=SITE).field)
    assert 4e-5 < np.linalg.norm(field) < 7e-5                     # tesla
    assert field[1] > 0 and field[2] < 0                           # north, and down
    local = np.ravel(Geomagnet(location=GRANDCS(x=0.0, y=0.0, z=0.0, location=SITE)).field)
    np.testing.assert_allclose(local, field)
    assert "tesla" in Geomagnet.__doc__ and "east-north-up" in Geomagnet.__doc__


def test_longitudes_are_stored_as_documented():
    assert float(np.ravel(Geodetic(latitude=40.0, longitude=-10.0, height=0.0).longitude)[0]) == 350.0
    with pytest.raises(ValueError, match="longitude"):            # stored as 400 before #267
        Geodetic(latitude=40.0, longitude=400.0, height=0.0)
    assert float(np.ravel(Geodetic(ECEF(Geodetic(latitude=40.0, longitude=350.0, height=0.0))).longitude)[0]) \
        == pytest.approx(350.0)
    assert "roughly corresponds to sea level" not in Geodetic.__doc__


@pytest.mark.parametrize("function", [grand.geoid_undulation, coordinates.geoid_undulation,
                                      topography.geoid_undulation])
def test_geoid_undulation_takes_the_same_arguments_everywhere(function):
    expected = coordinates.geoid_undulation(latitude=40.98, longitude=93.95)
    assert function(40.98, 93.95) == pytest.approx(expected)
    assert function(latitude=40.98, longitude=93.95) == pytest.approx(expected)
    assert float(np.ravel(function(SITE))[0]) == pytest.approx(expected)
    with pytest.warns(Warning):
        assert np.isnan(function()).all()               # as it always has (#262)
