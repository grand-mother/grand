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


def test_keyword_form_does_not_normalise_a_negative_longitude():
    r"""Records that ``longitude=-69.32`` returns ``nan`` where a `Geodetic` does not.

    The shipped EGM96 map is indexed over 0-360 degrees.  The
    ``latitude=``/``longitude=`` path passes the value through unchanged, so a
    western-hemisphere site gives ``nan``; passing a
    :class:`~grand.geo.coordinates.Geodetic` normalises and gives the right
    answer.  Anyone working at Auger, or anywhere else west of Greenwich, meets
    this immediately.

    Asserted rather than merely documented so that the day the keyword path
    normalises too, this test fails and the caveat can be removed from the
    documentation and from notebook 07.
    """
    lat, lon = AUGER
    negative = topography.geoid_undulation(latitude=lat, longitude=lon)
    wrapped = topography.geoid_undulation(latitude=lat, longitude=lon + 360.0)
    by_object = float(np.ravel(topography.geoid_undulation(
        Geodetic(latitude=lat, longitude=lon, height=0.0)))[0])

    assert np.isnan(negative), (
        'the keyword form now handles a negative longitude (%s); this test and '
        'the caveats that cite it are stale' % negative)
    assert np.isfinite(wrapped), 'wrapping into [0, 360) should work'
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


@needs_tiles
def test_ground_distance_shortens_near_the_horizon_over_rising_terrain():
    r"""Terrain is not a correction to the flat-ground path length.

    Over flat ground the distance to the ground grows as
    :math:`1/\cos\theta`.  Over real terrain that overestimates badly near the
    horizon, because the ground rises into the ray -- by a factor of nearly six
    at 89 degrees on the tile the tests ship against.  This is the property
    that makes :func:`grand.geo.topography.distance` worth calling at all, so
    it is asserted rather than left to notebook 07.
    """
    from grand.geo.coordinates import CartesianRepresentation

    name = _tiles()[0]
    lat0 = float(name[1:3]) * (1 if name[0] == 'N' else -1)
    lon0 = float(name[4:7]) * (1 if name[3] == 'E' else -1)
    ground = topography.elevation(
        Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5, height=0.0))
    origin = Geodetic(latitude=lat0 + 0.5, longitude=lon0 + 0.5,
                      height=ground + 1500.0)

    zenith = np.radians(89.0)
    direction = CartesianRepresentation(x=np.sin(zenith), y=0.0,
                                        z=-np.cos(zenith))
    real = float(np.ravel(topography.distance(origin, direction,
                                              maximum_distance=600e3))[0])
    flat = 1500.0 / np.cos(zenith)

    assert np.isfinite(real), 'the ray never reached the ground'
    assert real < flat, (
        'the terrain did not shorten the path: %.1f km against a flat-ground '
        '%.1f km' % (real / 1e3, flat / 1e3))
