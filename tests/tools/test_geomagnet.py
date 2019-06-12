# -*- coding: utf-8 -*-
"""
Unit tests for the grand.tools.geomagnet module
"""

import unittest

import grand.tools as tools
from grand.tools.coordinates import GeodeticRepresentation, LTP
from grand.tools.geomagnet import Geomagnet

import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation, ITRS


class GeomagnetTest(unittest.TestCase):
    """Unit tests for the geomagnet module"""


    def __init__(self, *args):
        super().__init__(*args)

        # The geo-magnetic field according to
        # http://geomag.nrcan.gc.ca/calc/mfcal-en.php"""
        self.ref = (0, 2.2983E-05, -4.0852E-05)

        # The corresponding Earth location
        self.location = EarthLocation(lat=45.0 * u.deg, lon=3.0 * u.deg,
                                      height=1000. * u.m)
        self.date = "2018-06-04"


    def assertField(self, field, tol=6):
        """Check that the magnetic field is consistent"""
        self.assertAlmostEqual((field.x / u.T).value, self.ref[0], tol)
        self.assertAlmostEqual((field.y / u.T).value, self.ref[1], tol)
        self.assertAlmostEqual((field.z / u.T).value, self.ref[2], tol)

    def get_coordinates(self, n=1, obstime=True):
        obstime = self.date if obstime else None
        if n == 1:
            return LTP(x=0 * u.m, y=0 * u.m, z=0 * u.m, location=self.location,
                       obstime=obstime)
        else:
            zero = n * (0 * u.m,)
            return LTP(x=zero, y=zero, z=zero, location=self.location,
                       obstime=obstime)

    def test_default(self):
        # Test the initialisation
        model = "IGRF12"
        self.assertEqual(tools.geomagnet.model(), model)

        # Test the default field getter
        c = self.get_coordinates()
        field = tools.geomagnet.field(c)
        self.assertEqual(field.x.size, 1)
        self.assertEqual(field.x.unit, u.T)
        self.assertEqual(field.y.unit, u.T)
        self.assertEqual(field.z.unit, u.T)
        self.assertEqual(c.obstime, field.obstime)
        self.assertField(field)

        # Test the vectorized getter
        n = 10
        c = self.get_coordinates(n)
        field = tools.geomagnet.field(c)
        self.assertEqual(field.x.size, n)
        self.assertEqual(c.obstime, field.obstime)
        for value in field:
            self.assertEqual(field.x.unit, u.T)
            self.assertEqual(field.y.unit, u.T)
            self.assertEqual(field.z.unit, u.T)
            frame = LTP(location=self.location)
            ltp = value.transform_to(frame)
            self.assertField(ltp)

        # Test the getter from ITRS
        itrs = ITRS(self.location.itrs.cartesian, obstime=self.date)
        field = tools.geomagnet.field(itrs)
        self.assertEqual(field.x.size, 1)
        self.assertEqual(c.obstime, field.obstime)
        self.assertField(field)


    def test_custom(self):
        geomagnet = Geomagnet(model="WMM2015")
        c = self.get_coordinates()
        field = geomagnet.field(c)
        self.assertEqual(field.x.size, 1)
        self.assertEqual(c.obstime, field.obstime)
        self.assertField(field)


    def test_error(self):
        c = self.get_coordinates(obstime=False)
        with self.assertRaises(ValueError) as context:
            tools.geomagnet.field(c)
        self.assertRegex(context.exception.args[0], "^No observation time")


if __name__ == "__main__":
    unittest.main()
