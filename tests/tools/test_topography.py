"""
Unit tests for the grand.geo.topography module
"""
import os
import unittest
from pathlib import Path

import numpy

# import astropy.units as u

import grand.store as store
import grand.geo as tools

# from grand.geo.coordinates import ECEF, GeodeticRepresentation, LTP
# from grand.geo import topography
# from grand.geo.topography import Topography
from tests import TestCase

from grand import Topography, geoid_undulation  # , Reference
from grand import topography
from grand import ECEF, Geodetic, LTP, GRANDCS


class TopographyTest(TestCase):
    """Unit tests for the topography module"""

    def test_geoid(self):
        # Test the undulation getter
        # c = ECEF(GeodeticRepresentation(latitude=45.5 * u.deg,
        #                                longitude=3.5 * u.deg))
        c = ECEF(Geodetic(latitude=45.5, longitude=3.5, height=0))
        # z = tools.topography.geoid_undulation(c)
        z = topography.geoid_undulation(c)
        self.assertEqual(z.size, 1)
        # self.assertEqual(z.unit, u.m)

    def test_custom_topography(self):
        # Fetch a test tile
        # dirname, basename = Path('tests/topography'), 'N39E090.SRTMGL1.hgt'
        dirname, basename = Path("../topography"), "N39E090.SRTMGL1.hgt"
        path = dirname / basename
        if not path.exists():
            dirname.mkdir(exist_ok=True)
            with path.open("wb") as f:
                f.write(store.get(basename))

        # Test the topography getter
        topo = Topography(dirname)
        # c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
        #                                longitude=90.5 * u.deg))
        c = ECEF(Geodetic(latitude=39.5, longitude=90.5, height=0))
        z = topo.elevation(c)
        self.assertEqual(z.size, 1)
        # self.assertEqual(z.unit, u.m)
        self.assertFalse(numpy.isnan(z))

    def test_topography_cache(self):
        # Check the cache config
        self.assertEqual(topography.model(), "SRTMGL1")
        self.assertRegex(str(topography.cachedir()), "^.*/grand/geo/data/topography")

        # Clear the cache
        topography.update_data(clear=True)
        self.assertFalse([p for p in topography.cachedir().glob("**/*")])

        # Fetch data
        # c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
        #                                longitude=90.5 * u.deg))
        c = ECEF(Geodetic(latitude=39.5, longitude=90.5, height=0))
        topography.update_data(c)
        self.assertTrue((topography.cachedir() / "N39E090.SRTMGL1.hgt").exists())

        # c = ECEF(GeodeticRepresentation(
        #    latitude=u.Quantity((39.5, 40.5)) * u.deg,
        #    longitude=u.Quantity((90.5, 90.5)) * u.deg))
        c = ECEF(
            Geodetic(
                latitude=numpy.array([39.5, 40.5]),
                longitude=numpy.array([90.5, 90.5]),
                height=numpy.array([0, 0]),
            )
        )
        topography.update_data(c)
        self.assertTrue((topography.cachedir() / "N40E090.SRTMGL1.hgt").exists())

        # c = ECEF(GeodeticRepresentation(latitude=40 * u.deg,
        #                                longitude=90.5 * u.deg))
        c = ECEF(Geodetic(latitude=40, longitude=90.5, height=0))
        # topography.update_data(c, radius=100 * u.km)
        topography.update_data(c, radius=100e3)  # RK radius in meters.
        self.assertTrue((topography.cachedir() / "N39E089.SRTMGL1.hgt").exists())
        self.assertTrue((topography.cachedir() / "N39E091.SRTMGL1.hgt").exists())
        self.assertTrue((topography.cachedir() / "N40E089.SRTMGL1.hgt").exists())
        self.assertTrue((topography.cachedir() / "N40E091.SRTMGL1.hgt").exists())

    def test_topography_elevation(self):
        # Fetch a test tile
        # geo = GeodeticRepresentation(latitude=39.5 * u.deg,
        #                             longitude=90.5 * u.deg)
        geo = Geodetic(latitude=39.5, longitude=90.5, height=0)
        c = ECEF(geo)
        topography.update_data(c)

        # Test the global topography getter
        # z0 = topography.elevation(c)
        z0 = topography.elevation(geo)
        self.assertEqual(z0.size, 1)
        #   self.assertEqual(z0.unit, u.m)
        self.assertFalse(numpy.isnan(z0))

        z1 = topography.elevation(geo, reference="ELLIPSOID")
        z1 -= topography.geoid_undulation(geo)
        self.assertEqual(z1.size, 1)
        # self.assertEqual(z1.unit, u.m)
        self.assertFalse(numpy.isnan(z1))

        self.assertQuantity(z0, z1)

        o = numpy.ones(10)
        # cv = ECEF(GeodeticRepresentation(latitude=geo.latitude * o,
        #                                 longitude=geo.longitude * o))
        cv = ECEF(
            Geodetic(
                latitude=geo.latitude * o,
                longitude=geo.longitude * o,
                height=geo.height * o,
            )
        )
        z2 = topography.elevation(cv)
        self.assertEqual(z2.size, o.size)
        # self.assertEqual(z2.unit, u.m)
        for i in range(o.size):
            self.assertQuantity(z2[i], z0)

        # Test the local topography getter
        # cl = LTP(0 << u.m, 0 << u.m, 0 << u.m, location=c)
        cl = LTP(x=0, y=0, z=0, location=c, orientation="NWU")
        z3 = topography.elevation(cl, "LOCAL")
        self.assertEqual(z3.size, 1)
        # self.assertEqual(z3.unit, u.m)
        self.assertQuantity(z3, z1, 7)

    def test_topography_distance(self):
        # Fetch a test tile
        # geo = GeodeticRepresentation(latitude=39.5 * u.deg,
        #                             longitude=90.5 * u.deg)
        # geo = Geodetic(latitude=39.5, longitude=90.5, height=0.)
        geo = Geodetic(latitude=41.27, longitude=96.53, height=0)
        # c = ECEF(geo)
        topography.update_data(geo)

        # Test the distance getter
        z = topography.elevation(geo) + topography.geoid_undulation(geo)
        # z = topography.elevation(c) + topography.geoid_undulation(c)
        # c = ECEF(GeodeticRepresentation(latitude=geo.latitude,
        #                                longitude=geo.longitude,
        #                                height=z - 1 * u.m))

        v0 = GRANDCS(x=100, y=10, z=-0.1, location=geo)
        x0 = GRANDCS(
            x=-8000, y=12000, z=2200, location=geo
        )  # Initial point along traj (in LTP frame)
        v = numpy.matmul(
            x0.basis.T, v0
        )  # GRANDCS --> ECEF. input direction must be in ECEF. ToDo: Fix this.

        # v = LTP(0 * u.deg, 45 * u.deg, representation_type='horizontal',
        #        location = c)

        # d = topography.distance(c, v, 10 * u.m)
        d = topography.distance(x0, v)  # (pos, dir, max_dist). max_dist is optional

        self.assertEqual(d.size, 1)
        # self.assertEqual(d.unit, u.m)
        self.assertFalse(numpy.isnan(d))
        self.assertTrue(d > 0)

        # d = topography.distance(c, v, 50 * u.cm)
        d = topography.distance(x0, v, 50)
        self.assertTrue(numpy.isnan(d))

        o = numpy.ones(10)
        # c = ECEF(GeodeticRepresentation(latitude=geo.latitude * o,
        #                                longitude=geo.longitude * o,
        #                                height=(z - 1 * u.m) * o))
        c = Geodetic(latitude=41.27 * o, longitude=96.53 * o, height=(z - 1) * o)
        # d = topography.distance(c, v, 10 * u.m)
        d = topography.distance(c, v * o)
        self.assertEqual(d.size, o.size)
        # self.assertEqual(d.unit, u.m)
        for i in range(o.size):
            self.assertTrue(d[i] < 0)


if __name__ == "__main__":
    unittest.main()
