"""
Unit tests for the grand.tools.geomagnet module
"""

import unittest

import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation, ITRS

import grand.tools as tools
from grand.tools.coordinates import GeodeticRepresentation, LTP
from grand.tools.geomagnet import Geomagnet
from tests import TestCase


class GeomagnetTest(TestCase):
    """Unit tests for the geomagnet module"""


    def __init__(self, *args):
        super().__init__(*args)

        # The geo-magnetic field according to
        # https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
        self.ref = (566E-09, 22999E-09, -41003E-09)

        # The corresponding Earth location
        self.location = EarthLocation(lat=45.0 * u.deg, lon=3.0 * u.deg,
                                      height=1000. * u.m)
        self.date = "2020-03-23"


    def assertField(self, field, tol=6):
        """Check that the magnetic field is consistent"""
        self.assertQuantity(field.x, self.ref[0] * u.T, tol)
        self.assertQuantity(field.y, self.ref[1] * u.T, tol)
        self.assertQuantity(field.z, self.ref[2] * u.T, tol)

    def get_coordinates(self, n=1, obstime=True):
        obstime = self.date if obstime else None
        if n == 1:
            return LTP(x=0 * u.m, y=0 * u.m, z=0 * u.m, location=self.location,
                       orientation="ENU", magnetic=False, obstime=obstime)
        else:
            zero = n * (0 * u.m,)
            return LTP(x=zero, y=zero, z=zero, location=self.location,
                       orientation="ENU", magnetic=False, obstime=obstime)

    def test_default(self):
        # Test the initialisation
        self.assertEqual(tools.geomagnet.model,
                         tools.geomagnet._default_model)
        self.assertEqual(tools.geomagnet.obstime,
                         tools.geomagnet._default_obstime.datetime.date)

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
            frame = LTP(location=self.location, orientation="ENU",
                        magnetic=False)
            ltp = value.transform_to(frame)
            self.assertField(ltp)

        # Test the getter from ITRS
        itrs = ITRS(self.location.itrs.cartesian, obstime=self.date)
        field = tools.geomagnet.field(itrs)
        self.assertEqual(field.x.size, 1)
        self.assertEqual(c.obstime, field.obstime)
        self.assertField(field)


    def test_custom(self):
        geomagnet = Geomagnet(model="WMM2020")
        c = self.get_coordinates()
        field = geomagnet.field(c)
        self.assertEqual(field.x.size, 1)
        self.assertEqual(c.obstime, field.obstime)
        self.assertField(field)


    def test_obstime(self):
        c = self.get_coordinates(obstime=False)
        field = tools.geomagnet.field(c)
        self.assertEqual(field.obstime.datetime.date,
                         tools.geomagnet._default_obstime.datetime.date)


if __name__ == "__main__":
    unittest.main()
