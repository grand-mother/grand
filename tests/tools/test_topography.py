# -*- coding: utf-8 -*-
"""
Unit tests for the grand.tools.topography module
"""

import os
import unittest
from pathlib import Path

import grand.store as store
import grand.tools as tools
from grand.tools.coordinates import ECEF, GeodeticRepresentation
from grand.tools import topography
from grand.tools.topography import Topography

import numpy
import astropy.units as u


class TopographyTest(unittest.TestCase):
    """Unit tests for the topography module"""


    def test_geoid(self):
        # Test the undulation getter
        c = ECEF(GeodeticRepresentation(latitude=45.5 * u.deg,
                                        longitude=3.5 * u.deg))
        z = tools.topography.geoid_undulation(c)
        self.assertEqual(z.size, 1)
        self.assertEqual(z.unit, u.m)


    def test_custom_topography(self):
        # Fetch a test tile
        dirname, basename = Path("tests/topography"), "N39E090.SRTMGL1.hgt"
        path = dirname / basename
        if not path.exists():
            dirname.mkdir(exist_ok=True)
            with path.open("wb") as f:
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
        self.assertEqual(topography.model(), "SRTMGL1")
        self.assertRegex(str(topography.cachedir()),
                "^.*/grand/tools/data/topography")

        # Clear the cache
        topography.update_data(clear=True)
        self.assertFalse([p for p in topography.cachedir().glob("**/*")])

        # Fetch data
        c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
                                        longitude=90.5 * u.deg))
        topography.update_data(c)
        self.assertTrue(
            (topography.cachedir() / "N39E090.SRTMGL1.hgt").exists())

        c = ECEF(GeodeticRepresentation(
            latitude=u.Quantity((39.5, 40.5)) * u.deg,
            longitude=u.Quantity((90.5, 90.5)) * u.deg))
        topography.update_data(c)
        self.assertTrue(
            (topography.cachedir() / "N40E090.SRTMGL1.hgt").exists())

        c = ECEF(GeodeticRepresentation(latitude=40 * u.deg,
                                        longitude=90.5 * u.deg))
        topography.update_data(c, radius=100 * u.km)
        self.assertTrue(
            (topography.cachedir() / "N39E089.SRTMGL1.hgt").exists())
        self.assertTrue(
            (topography.cachedir() / "N39E091.SRTMGL1.hgt").exists())
        self.assertTrue(
            (topography.cachedir() / "N40E089.SRTMGL1.hgt").exists())
        self.assertTrue(
            (topography.cachedir() / "N40E091.SRTMGL1.hgt").exists())


    def test_topography_elevation(self):
        # Fetch a test tile
        c = ECEF(GeodeticRepresentation(latitude=39.5 * u.deg,
                                        longitude=90.5 * u.deg))
        topography.update_data(c)

        # Test the topography getter
        z = topography.elevation(c)
        self.assertEqual(z.size, 1)
        self.assertEqual(z.unit, u.m)
        self.assertFalse(numpy.isnan(z))


if __name__ == "__main__":
    unittest.main()
