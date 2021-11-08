'''
Unit tests for the grand.tools.geomagnet module
'''

import unittest

import numpy
import datetime
#import astropy.units as u
#from astropy.coordinates import EarthLocation, ITRS

import grand.tools as tools
from grand.tools.coordinates import GeodeticRepresentation, LTP, \
        Geodetic, CartesianRepresentation, ECEF
from grand.tools.geomagnet import Geomagnet
from tests import TestCase


class GeomagnetTest(TestCase):
    '''Unit tests for the geomagnet module'''


    def __init__(self, *args):
        super().__init__(*args)

        # The geo-magnetic field according to
        # https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
        #self.ref = (566E-09, 22999E-09, -41003E-09)
        self.ref = CartesianRepresentation(x=566E-09, y=22999E-09, z=-41003E-09)

        # The corresponding Earth location
        #self.location = EarthLocation(lat=45.0 * u.deg, lon=3.0 * u.deg,
        #                              height=1000. * u.m)
        self.location = Geodetic(latitude=45.0, longitude=3.0, height=1000.)
        self.date = datetime.date(2020, 1, 1)


    def assertField(self, field, tol=6):
        '''Check that the magnetic field is consistent'''
        #self.assertQuantity(field.x, self.ref[0] * u.T, tol)
        #self.assertQuantity(field.y, self.ref[1] * u.T, tol)
        #self.assertQuantity(field.z, self.ref[2] * u.T, tol)
        self.assertQuantity(field.x, self.ref.x, tol)
        self.assertQuantity(field.y, self.ref.y, tol)
        self.assertQuantity(field.z, self.ref.z, tol)

    def get_coordinates(self, n=1, obstime=True):
        obstime = self.date if obstime else None
        if n == 1:
            #return LTP(x=0 * u.m, y=0 * u.m, z=0 * u.m, location=self.location,
            #           orientation='ENU', magnetic=False, obstime=obstime)
            return LTP(x=0, y=0, z=0, location=self.location,
                       orientation='ENU', magnetic=False, obstime=obstime)
        else:
            #zero = n * (0 * u.m,)
            zero = numpy.asarray(n * [0])
            return LTP(x=zero, y=zero, z=zero, location=self.location,
                       orientation='ENU', magnetic=False, obstime=obstime)

    def test_default(self):
        # Test the initialisation
        self.assertEqual(tools.geomagnet.model,
                         tools.geomagnet._default_model)
        #self.assertEqual(tools.geomagnet.obstime,
        #                 tools.geomagnet._default_obstime.datetime.date)
        self.assertEqual(tools.geomagnet.obstime,
                         tools.geomagnet._default_obstime)

        # Test the default field getter
        c = self.get_coordinates()
        field = tools.geomagnet.field(c)
        self.assertEqual(field.x.size, 1)
        #self.assertEqual(field.x.unit, u.T)
        #self.assertEqual(field.y.unit, u.T)
        #self.assertEqual(field.z.unit, u.T)
        #self.assertEqual(c.obstime, field.obstime)
        self.assertEqual(c.obstime, tools.geomagnet.obstime)
        self.assertField(field)

        # Test the vectorized getter
        n = 10
        c = self.get_coordinates(n)
        field = tools.geomagnet.field(c)
        self.assertEqual(field.x.size, n)
        self.assertEqual(c.obstime, tools.geomagnet.obstime)
        for i in range(len(field.x)):
            value = field[:,i]
            value = CartesianRepresentation(x=value[0], y=value[1], z=value[2])
            self.assertField(value)
        #for value in field:
        #    #self.assertEqual(field.x.unit, u.T)
        #    #self.assertEqual(field.y.unit, u.T)
        #    #self.assertEqual(field.z.unit, u.T)
        # RK: Not sure what is done here.
        #    frame = LTP(location=self.location, orientation='ENU',
        #                magnetic=False)
        #    ltp = value.transform_to(frame)
        #    self.assertField(ltp)

        # Test the getter from ITRS
        #itrs = ITRS(self.location.itrs.cartesian, obstime=self.date)
        #field = tools.geomagnet.field(itrs)
        ecef  = ECEF(self.location)
        field = tools.geomagnet.field(ecef)
        self.assertEqual(field.x.size, 1)
        #self.assertEqual(c.obstime, field.obstime)
        self.assertEqual(c.obstime, tools.geomagnet.obstime)
        self.assertField(field)


    def test_custom(self):
        c = self.get_coordinates()
        geomagnet = Geomagnet(model='WMM2020', location=c)
        #c = self.get_coordinates()
        #field = geomagnet.field(c)
        field = geomagnet.field
        self.assertEqual(field.x.size, 1)
        #self.assertEqual(c.obstime, field.obstime)
        self.assertEqual(c.obstime, geomagnet.obstime)
        self.assertField(field)


    def test_obstime(self):
        c = self.get_coordinates(obstime=False)
        #field = tools.geomagnet.field(c)
        geomagnet = Geomagnet(location=c)
        #self.assertEqual(field.obstime.datetime.date,
        #                 tools.geomagnet._default_obstime.datetime.date)
        self.assertEqual(geomagnet.obstime,
                         tools.geomagnet.obstime)


if __name__ == '__main__':
    unittest.main()
