'''
Unit tests for the grand.tools.topography module
'''

import os
import unittest
from pathlib import Path

import numpy
import astropy.units as u

import grand.store as store
import grand.tools as tools
from grand.tools.coordinates import ECEF, GeodeticRepresentation, LTP
from grand.tools import topography
from grand.tools.topography import Topography
from tests import TestCase


class TopographyTest(TestCase):
    '''Unit tests for the topography module'''


    def test_geoid(self):
        # Test the undulation getter
        c = ECEF(GeodeticRepresentation(latitude=45.5 * u.deg,
                                        longitude=3.5 * u.deg))
        z = tools.topography.geoid_undulation(c)
        self.assertEqual(z.size, 1)
        self.assertEqual(z.unit, u.m)


    def test_custom_topography(self):
        # Fetch a test tile
        dirname, basename = Path('tests/topography'), 'N39E090.SRTMGL1.hgt'
        path = dirname / basename
        if not path.exists():
            dirname.mkdir(exist_ok=True)
            with path.open('wb') as f:
                f.write(store.get(basename))

        # Test the topography getter
        topo = Topography(dirname)
        c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
                                        longitude=90.5 * u.deg))
        z = topo.elevation(c)
        self.assertEqual(z.size, 1)
        self.assertEqual(z.unit, u.m)
        self.assertFalse(numpy.isnan(z))


    def test_topography_cache(self):
        # Check the cache config
        self.assertEqual(topography.model(), 'SRTMGL1')
        self.assertRegex(str(topography.cachedir()),
                '^.*/grand/tools/data/topography')

        # Clear the cache
        topography.update_data(clear=True)
        self.assertFalse([p for p in topography.cachedir().glob('**/*')])

        # Fetch data
        c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
                                        longitude=90.5 * u.deg))
        topography.update_data(c)
        self.assertTrue(
            (topography.cachedir() / 'N39E090.SRTMGL1.hgt').exists())

        c = ECEF(GeodeticRepresentation(
            latitude=u.Quantity((39.5, 40.5)) * u.deg,
            longitude=u.Quantity((90.5, 90.5)) * u.deg))
        topography.update_data(c)
        self.assertTrue(
            (topography.cachedir() / 'N40E090.SRTMGL1.hgt').exists())

        c = ECEF(GeodeticRepresentation(latitude=40 * u.deg,
                                        longitude=90.5 * u.deg))
        topography.update_data(c, radius=100 * u.km)
        self.assertTrue(
            (topography.cachedir() / 'N39E089.SRTMGL1.hgt').exists())
        self.assertTrue(
            (topography.cachedir() / 'N39E091.SRTMGL1.hgt').exists())
        self.assertTrue(
            (topography.cachedir() / 'N40E089.SRTMGL1.hgt').exists())
        self.assertTrue(
            (topography.cachedir() / 'N40E091.SRTMGL1.hgt').exists())


    def test_topography_elevation(self):
        # Fetch a test tile
        geo = GeodeticRepresentation(latitude=39.5 * u.deg,
                                     longitude=90.5 * u.deg)
        c = ECEF(geo)
        topography.update_data(c)

        # Test the global topography getter
        z0 = topography.elevation(c)
        self.assertEqual(z0.size, 1)
        self.assertEqual(z0.unit, u.m)
        self.assertFalse(numpy.isnan(z0))

        z1 = topography.elevation(c, reference=topography.Reference.GEOID)
        z1 += topography.geoid_undulation(c)
        self.assertEqual(z1.size, 1)
        self.assertEqual(z1.unit, u.m)
        self.assertFalse(numpy.isnan(z1))

        self.assertQuantity(z0, z1)

        o = numpy.ones(10)
        cv = ECEF(GeodeticRepresentation(latitude=geo.latitude * o,
                                         longitude=geo.longitude * o))
        z2 = topography.elevation(cv)
        self.assertEqual(z2.size, o.size)
        self.assertEqual(z2.unit, u.m)
        for i in range(o.size):
            self.assertQuantity(z2[i], z0)

        # Test the local topography getter
        cl = LTP(0 << u.m, 0 << u.m, 0 << u.m, location=c)
        z3 = topography.elevation(cl)
        self.assertEqual(z3.size, 1)
        self.assertEqual(z3.unit, u.m)
        self.assertQuantity(z3, z1, 7)


    def test_topography_distance(self):
        # Fetch a test tile
        geo = GeodeticRepresentation(latitude=39.5 * u.deg,
                                     longitude=90.5 * u.deg)
        c = ECEF(geo)
        topography.update_data(c)

        # Test the distance getter
        z = topography.elevation(c) + topography.geoid_undulation(c)
        c = ECEF(GeodeticRepresentation(latitude=geo.latitude,
                                        longitude=geo.longitude,
                                        height=z - 1 * u.m))
        v = LTP(0 * u.deg, 45 * u.deg, representation_type='horizontal',
                location = c)
        d = topography.distance(c, v, 10 * u.m)
        self.assertEqual(d.size, 1)
        self.assertEqual(d.unit, u.m)
        self.assertFalse(numpy.isnan(d))
        self.assertTrue(d < 0)

        d = topography.distance(c, v, 50 * u.cm)
        self.assertTrue(numpy.isnan(d))

        o = numpy.ones(10)
        c = ECEF(GeodeticRepresentation(latitude=geo.latitude * o,
                                        longitude=geo.longitude * o,
                                        height=(z - 1 * u.m) * o))
        d = topography.distance(c, v, 10 * u.m)
        self.assertEqual(d.size, o.size)
        self.assertEqual(d.unit, u.m)
        for i in range(o.size):
            self.assertTrue(d[i] < 0)


if __name__ == '__main__':
    unittest.main()
