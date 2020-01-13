"""
Unit tests for the grand_tools.coordinates module
"""

import unittest

import numpy
import astropy.units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, ITRS,  \
                                SkyCoord
from astropy.utils import iers
iers.conf.auto_download = False # Disable downloads for tests,
                                # due to latency failures

from grand.tools.coordinates import ECEF, LTP, GeodeticRepresentation,         \
                                    HorizontalRepresentation
from tests import TestCase


class CoordinatesTest(TestCase):
    """Unit tests for the coordinates module"""

    def __init__(self, *args):
        super().__init__(*args)

        self.obstime = "2010-01-01"
        self.location = EarthLocation(lat=45.0 * u.deg, lon=3.0 * u.deg,
                                      height=1000. * u.m)

    def test_geodetic(self):
        cart = self.location.itrs.cartesian

        # Check the Cartesian generator
        r = GeodeticRepresentation.from_cartesian(cart)

        self.assertQuantity(r.latitude, self.location.lat, 11)
        self.assertQuantity(r.longitude, self.location.lon, 11)
        self.assertQuantity(r.height, self.location.height, 6)

        # Check the Cartesian conversion
        r = GeodeticRepresentation(self.location.lat, self.location.lon,
                                   self.location.height).to_cartesian()

        self.assertCartesian(r, cart, 6)

        # Vectorize the test point
        n = 10
        vectorize = lambda v: u.Quantity(
            numpy.repeat((v[0] / v[1]).value, n), v[1])
        x, y, z = map(vectorize, ((cart.x, u.m), (cart.y, u.m), (cart.z, u.m)))
        cart = CartesianRepresentation(x=x, y=y, z=z)
        latitude, longitude, height = map(vectorize, (
            (self.location.lat, u.deg), (self.location.lon, u.deg),
            (self.location.height, u.m)))

        # Check the vectorized Cartesian generator
        r = GeodeticRepresentation.from_cartesian(cart)

        for i in range(n):
            self.assertQuantity(r.latitude[i], self.location.lat, 11)
            self.assertQuantity(r.longitude[i], self.location.lon, 11)
            self.assertQuantity(r.height[i], self.location.height, 6)

        # Check the vetcorized Cartesian conversion
        r = GeodeticRepresentation(latitude, longitude, height).to_cartesian()
        self.assertCartesian(r, cart, 6)


    def test_horizontal(self):
        for (angle, point) in (((90, 0), (1, 0, 0)),
                               (( 0, 0), (0, 1, 0)),
                               (( 0, 90), (0, 0, 1)),
                               (( -90, 0), (-1, 0, 0))):
            h = HorizontalRepresentation(azimuth=angle[0] * u.deg,
                                         elevation=angle[1] * u.deg)
            cart = h.represent_as(CartesianRepresentation)

            self.assertQuantity(cart.x, point[0] * u.one, 9)
            self.assertQuantity(cart.y, point[1] * u.one, 9)
            self.assertQuantity(cart.z, point[2] * u.one, 9)

            cart = CartesianRepresentation(*point)
            h = cart.represent_as(HorizontalRepresentation)

            self.assertQuantity(h.azimuth, angle[0] * u.deg, 7)
            self.assertQuantity(h.elevation, angle[1] * u.deg, 7)


    def test_ecef(self):
        # Check the forward transform
        ecef = ECEF(self.location.itrs.cartesian, obstime=self.obstime)
        itrs = ecef.transform_to(ITRS(obstime=self.obstime))
        self.assertCartesian(ecef, itrs, 8)

        # Check the backward transform
        ecef0 = itrs.transform_to(ECEF(obstime=self.obstime))
        self.assertCartesian(ecef, ecef0, 8)

        # Check the obstime handling
        ecef1 = itrs.transform_to(ECEF)
        self.assertEqual(ecef1.obstime, itrs.obstime)
        self.assertCartesian(ecef1, ecef0, 8)

        # Check the round trip with different obstimes
        itrs = ecef.transform_to(ITRS)
        ecef0 = itrs.transform_to(ECEF(obstime=self.obstime))
        self.assertCartesian(ecef, ecef0, 2)

        # Check the Earth location conversion
        location = ecef0.earth_location.itrs.cartesian
        self.assertCartesian(ecef, location, 2)


    def test_ltp(self):
        ecef = ECEF(self.location.itrs.cartesian, obstime=self.obstime)

        # Check the constructor & to ECEF transform
        ltp = LTP(x=0 * u.m, y=0 * u.m, z=0 * u.m, location=self.location) 
        r = ltp.transform_to(ECEF)

        self.assertEqual(r.obstime, ltp.obstime)
        self.assertCartesian(r, ecef, 6)

        # Check the from ECEF transform
        ltp = ecef.transform_to(LTP(location=self.location))

        self.assertEqual(ltp.obstime, self.obstime)
        self.assertQuantity(ltp.x, 0 * u.m, 6)
        self.assertQuantity(ltp.y, 0 * u.m, 6)
        self.assertQuantity(ltp.z, 0 * u.m, 6)

        # Check the Earth location conversion
        location = ltp.earth_location.itrs.cartesian
        self.assertCartesian(ecef, location, 2)

        # Check the affine transform
        points = ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1),
                  (0, 1, 1))
        for point in (points):
            cart = CartesianRepresentation(x=point[1], y=point[0], z=point[2],
                                           unit=u.m)
            altaz = SkyCoord(cart, frame="altaz", location=self.location,
                             obstime=self.obstime)
            ecef0 = altaz.transform_to(ECEF(obstime=self.obstime))

            cart = CartesianRepresentation(x=point[0], y=point[1], z=point[2],
                                           unit=u.m)
            ltp = LTP(cart, location=self.location, obstime=self.obstime)
            ecef1 = ltp.transform_to(ECEF)

            self.assertEqual(ecef0.obstime, ecef1.obstime)
            self.assertCartesian(ecef0, ecef1, 4)

        # Check the orientation
        point = (1, -1, 2)
        cart = CartesianRepresentation(x=point[0], y=point[1], z=point[2],
                                       unit=u.m)
        altaz = SkyCoord(cart, frame="altaz", location=self.location,
                         obstime=self.obstime)
        ecef0 = altaz.transform_to(ECEF(obstime=self.obstime))

        for (orientation, sign) in ((("N", "E", "U"), (1, 1, 1)),
                                    (("N", "E", "D"), (1, 1, -1)),
                                    (("S", "E", "U"), (-1, 1, 1)),
                                    (("N", "W", "U"), (1, -1, 1))):
            cart = CartesianRepresentation(x=sign[0] * point[0],
                y=sign[1] * point[1], z=sign[2] * point[2], unit=u.m)
            ltp = LTP(cart, location=self.location, obstime=self.obstime,
                      orientation=orientation)
            ecef1 = ltp.transform_to(ECEF(obstime=self.obstime))

            self.assertCartesian(ecef0, ecef1, 4)

        # Check the unit vector case
        uy = HorizontalRepresentation(azimuth = 0 * u.deg,
                                      elevation = 0 * u.deg)
        ltp = LTP(uy, location=self.location, obstime=self.obstime)

        self.assertQuantity(ltp.x, 0 * u.one, 9)
        self.assertQuantity(ltp.y, 1 * u.one, 9)
        self.assertQuantity(ltp.z, 0 * u.one, 9)

        r = ltp.transform_to(ECEF)

        self.assertEqual(r.obstime, ltp.obstime)
        self.assertQuantity(r.cartesian.norm(), 1 * u.one, 6)

        ecef = ECEF(uy, obstime=self.obstime)
        ltp = ecef.transform_to(LTP(location=self.location))

        self.assertEqual(ltp.obstime, ecef.obstime)
        self.assertQuantity(ltp.cartesian.norm(), 1 * u.one, 6)

        # Check the magnetic north case
        ltp0 = LTP(uy, location=self.location, obstime=self.obstime)
        frame1 = LTP(location=self.location, obstime=self.obstime,
                     magnetic=True)
        ltp1 = ltp0.transform_to(frame1)
        self.assertEqual(ltp0.obstime, ltp1.obstime)

        declination = numpy.arcsin(ltp0.cartesian.cross(ltp1.cartesian).norm())
        self.assertQuantity(declination.to(u.deg), 0.10 * u.deg, 2)

        # Test the magnetic case with no obstime
        with self.assertRaises(ValueError) as context:
            LTP(uy, location=self.location, magnetic=True)
        self.assertRegex(context.exception.args[0], "^Magnetic")

        # Test the invalid frame case
        with self.assertRaises(ValueError) as context:
            LTP(uy, location=self.location, orientation=("T", "O", "T", "O"))
        self.assertRegex(context.exception.args[0], "^Invalid frame")

        # Test the ltp round trip with a position
        frame0 = LTP(location=self.location)
        frame1 = LTP(location=self.location, obstime=self.obstime,
                     magnetic=True)
        ltp0 = LTP(x=1 * u.m, y=2 * u.m, z=3 * u.m, location=self.location,
                   obstime=self.obstime)
        ltp1 = ltp0.transform_to(frame1).transform_to(frame0)
        self.assertCartesian(ltp0, ltp1, 8)

        # Test the same frame case
        ltp1 = ltp0.transform_to(frame0)
        self.assertCartesian(ltp0, ltp1, 8)
        self.assertEqual(ltp0.obstime, ltp1.obstime)

        # Test an LTP permutation
        ltp0 = LTP(x=1 * u.m, y=2 * u.m, z=3 * u.m, location=self.location)
        frame1 = LTP(location=self.location, orientation=("N", "E", "D"))
        ltp1 = ltp0.transform_to(frame1)
        self.assertQuantity(ltp0.x, ltp1.y, 6)
        self.assertQuantity(ltp0.y, ltp1.x, 6)
        self.assertQuantity(ltp0.z, -ltp1.z, 6)


if __name__ == "__main__":
    unittest.main()
