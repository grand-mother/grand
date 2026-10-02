# -*- coding: utf-8 -*-
r"""Pins the conventions and the silent failures of :mod:`grand.geo.topography`.

Two of these are traps rather than bugs in the usual sense: the function
returns ``nan`` rather than raising, so a mistake propagates into a geometry
calculation and stays plausible for several steps.  They are asserted here so
that the behaviour is at least written down, and so that a future fix is
noticed rather than silently changing results.

Elevation lookups need SRTM tiles, which ``data/.gitignore`` excludes from
version control.  Those tests skip when no tile is present rather than fail, so
this file is meaningful in CI without a download.  The geoid tests always run:
``data/egm96.png`` *is* tracked.
"""

import os

import numpy as np
import pytest

from grand.geo import topography
from grand.geo.coordinates import Geodetic

#: A site in the western hemisphere, where the longitude convention matters.
AUGER = (-35.20, -69.32)

#: A site in the eastern hemisphere, where it does not.
GP300 = (40.98, 93.95)


def _tiles():
    r"""Returns the SRTM tiles available on this machine.

    Returns
    -------
    list of str
        File names, possibly empty.
    """
    datadir = str(topography.datadir())
    if not os.path.isdir(datadir):
        return []
    return sorted(f for f in os.listdir(datadir) if f.endswith('.hgt'))


needs_tiles = pytest.mark.skipif(
    not _tiles(), reason='no SRTM tiles present; see topography.update_data')


# --------------------------------------------------------------------------
# the geoid, which ships with the package
# --------------------------------------------------------------------------

def test_geoid_undulation_is_finite_in_the_eastern_hemisphere():
    r"""The straightforward case works through either calling convention."""
    lat, lon = GP300
    by_keyword = topography.geoid_undulation(latitude=lat, longitude=lon)
    by_object = topography.geoid_undulation(
        Geodetic(latitude=lat, longitude=lon, height=0.0))
    assert np.isfinite(by_keyword), 'geoid undulation is nan at the GP300 site'
    assert np.allclose(np.ravel(by_object)[0], by_keyword), (
        'the two calling conventions disagree: %s vs %s'
        % (np.ravel(by_object)[0], by_keyword))


def test_keyword_form_normalises_a_negative_longitude():
    r"""``longitude=-69.32`` gives the same undulation as a `Geodetic` (#251).

    The shipped EGM96 map is indexed over 0-360 degrees.  The
    ``latitude=``/``longitude=`` path passed the value through unchanged, so a
    western-hemisphere site gave ``nan``; it now wraps the longitude, like the
    :class:`~grand.geo.coordinates.Geodetic` path.
    """
    lat, lon = AUGER
    negative = topography.geoid_undulation(latitude=lat, longitude=lon)
    wrapped = topography.geoid_undulation(latitude=lat, longitude=lon + 360.0)
    by_object = float(np.ravel(topography.geoid_undulation(
        Geodetic(latitude=lat, longitude=lon, height=0.0)))[0])

    assert np.isfinite(negative), 'a negative longitude gave %s' % negative
    assert np.isclose(negative, wrapped)
    assert np.isclose(wrapped, by_object), (
        'the Geodetic form disagrees with the wrapped keyword form: %s vs %s'
        % (by_object, wrapped))


def test_geodetic_form_works_across_the_whole_globe():
    r"""Every hemisphere gives a finite undulation through the `Geodetic` path.

    This is the calling convention the documentation recommends, so it is the
    one that has to hold everywhere.
    """
    for lat, lon in [(40.98, 93.95), (-35.20, -69.32), (72.0, -40.0),
                     (0.0, 78.0), (-60.0, 170.0), (10.0, -170.0)]:
        value = float(np.ravel(topography.geoid_undulation(
            Geodetic(latitude=lat, longitude=lon, height=0.0)))[0])
        assert np.isfinite(value), (
            'geoid undulation is nan at lat %.2f, lon %.2f' % (lat, lon))
        assert abs(value) < 120.0, (
            'geoid undulation of %.1f m at lat %.2f, lon %.2f is outside the '
            'physical range of about +/-110 m' % (value, lat, lon))


# --------------------------------------------------------------------------
# elevation, which needs tiles
# --------------------------------------------------------------------------

def test_elevation_returns_nan_where_no_tile_is_present():
    r"""Records that a missing tile is a ``nan``, not an exception.

    This is the most common way a topography calculation goes wrong: nothing
    raises, and the ``nan`` reaches the result several steps later.  It also
    propagates through the sea-level reference, which subtracts the undulation
    from it.
    """
    # A one-degree square in the middle of the Pacific, which nobody downloads.
    nowhere = Geodetic(latitude=-30.0, longitude=210.0, height=0.0)
    assert np.isnan(topography.elevation(nowhere)), (
        'a missing tile now raises or returns a value; this test is stale')
    assert np.isnan(topography.elevation(nowhere, reference='sea')), (
        'the nan no longer propagates through the sea-level reference')


@needs_tiles
def test_elevation_is_finite_inside_an_available_tile():
    r"""Inside a downloaded square the elevation is finite and plausible."""
    name = _tiles()[0]
    lat0 = float(name[1:3]) * (1 if name[0] == 'N' else -1)
    lon0 = float(name[4:7]) * (1 if name[3] == 'E' else -1)
    centre = Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5, height=0.0)

    value = topography.elevation(centre)
    assert np.isfinite(value), 'elevation is nan at the centre of tile %s' % name
    assert -500.0 < value < 9000.0, (
        'elevation of %.1f m in tile %s is outside the range of the Earth'
        % (value, name))


@needs_tiles
def test_elevation_is_vectorised():
    r"""A `Geodetic` holding arrays gives one value per point.

    Worth pinning: the vectorised path is a single TURTLE call where a loop is
    one per point, and notebook 07 depends on it to map a whole tile.
    """
    name = _tiles()[0]
    lat0 = float(name[1:3]) * (1 if name[0] == 'N' else -1)
    lon0 = float(name[4:7]) * (1 if name[3] == 'E' else -1)

    n = 16
    lats = np.linspace(lat0 + 0.1, lat0 + 0.9, n)
    lons = np.full(n, lon0 + 0.5)
    values = topography.elevation(
        Geodetic(latitude=lats, longitude=lons, height=np.zeros(n)))

    assert np.shape(values) == (n,), 'expected one elevation per point, got %s' % (
        np.shape(values),)
    assert np.isfinite(values).all(), 'a point inside the tile came back nan'


def _start_above_tile_centre(height):
    name = _tiles()[0]
    lat0 = float(name[1:3]) * (1 if name[0] == 'N' else -1)
    lon0 = float(name[4:7]) * (1 if name[3] == 'E' else -1)
    ground = topography.elevation(
        Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5, height=0.0))
    return Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5,
                    height=float(np.ravel(ground)[0]) + height)


def _down(zenith_deg):
    from grand.geo.coordinates import CartesianRepresentation

    th = np.radians(zenith_deg)
    return CartesianRepresentation(x=np.sin(th), y=0.0, z=-np.cos(th))


@needs_tiles
def test_straight_down_in_the_local_frame_gives_the_height():
    r"""#210: an (east, north, up) direction was read as ECEF.

    Straight down from 1500 m above the terrain gave 2268 m.  With
    ``frame="ENU"``, or the matching `LTP`, it is 1500 m.
    """
    from grand.geo.coordinates import LTP

    origin = _start_above_tile_centre(1500.0)
    by_name = float(np.ravel(topography.distance(origin, _down(0.0), 600e3, frame="ENU"))[0])
    by_ltp = float(np.ravel(topography.distance(
        origin, _down(0.0), 600e3, frame=LTP(location=origin, orientation="ENU", magnetic=False)))[0])
    assert by_name == pytest.approx(1500.0, abs=1.0)
    assert by_ltp == pytest.approx(by_name)


@needs_tiles
def test_terrain_changes_the_inclined_path_by_a_few_per_cent():
    r"""#210: the terrain correction at this site is small, not a factor of six.

    Over flat ground the distance grows as :math:`1/\cos\theta`.  On the tile
    the tests ship against, the real path is within ten per cent of that at
    45 and 80 degrees.
    """
    origin = _start_above_tile_centre(1500.0)
    for zenith in (10.0, 45.0, 80.0):
        real = float(np.ravel(topography.distance(origin, _down(zenith), 600e3, frame="ENU"))[0])
        flat = 1500.0 / np.cos(np.radians(zenith))
        assert np.isfinite(real)
        assert abs(real / flat - 1) < 0.1, (zenith, real, flat)


def test_an_unknown_frame_is_refused():
    with pytest.raises(ValueError, match="frame must be"):
        topography.distance(Geodetic(latitude=41.5, longitude=96.5, height=3000.0),
                            _down(0.0), frame="NED")
