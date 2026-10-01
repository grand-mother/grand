# -*- coding: utf-8 -*-
r"""#251: geodetic heights west of Greenwich; Horizontal locations; empty methods."""

import numpy as np
import pytest

from grand import ECEF, GRANDCS, LTP, Geodetic, Horizontal


def test_geoid_height_is_finite_west_of_greenwich():
    r"""The geoid map spans longitudes 0-360; a negative longitude gave NaN."""
    for longitude in (-69.3, 290.7):
        g = Geodetic(ECEF(Geodetic(latitude=-35.2, longitude=longitude, height=1400.0)))
        assert g.height[0] == pytest.approx(1400.0, abs=1e-3)


def test_a_second_horizontal_does_not_move_the_first():
    s1 = Geodetic(latitude=40.0, longitude=90.0, height=0.0)
    s2 = Geodetic(latitude=-35.0, longitude=-69.0, height=0.0)
    h1 = Horizontal(azimuth=90.0, elevation=0.0, norm=1000.0, location=s1)
    before = np.array(h1.horizontal_to_ecef()).copy()
    Horizontal(azimuth=90.0, elevation=0.0, norm=1000.0, location=s2)
    assert np.allclose(np.array(h1.horizontal_to_ecef()), before)


def test_conversions_return_values():
    from grand.basis.validate import GRANDlibWarning

    site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
    point = LTP(x=100.0, y=0.0, z=0.0, location=site, orientation="NWU")
    gcs = point.ltp_to_grandcs(location=site)
    assert isinstance(gcs, GRANDCS)
    assert np.allclose(np.ravel(gcs), np.ravel(GRANDCS(point, location=site)))
    with pytest.warns(GRANDlibWarning, match="default array origin"):
        assert point.ltp_to_grandcs() is not None
    far = Geodetic(ECEF(point))
    assert isinstance(far.geodetic_to_horizontal(site), Horizontal)
    with pytest.raises(TypeError, match="give the 'location'"):
        far.geodetic_to_horizontal()
