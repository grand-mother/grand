'''
Unit tests for the grand_tools.coordinates module
'''

import unittest

import numpy
#import astropy.units as u
#from astropy.coordinates import CartesianRepresentation, EarthLocation, ITRS,  \
#                                SkyCoord, SphericalRepresentation
#from astropy.utils import iers
iers.conf.auto_download = False # Disable downloads for tests,
                                # due to latency failures

from grand import ECEF, LTP, Geodetic, GeodeticRepresentation, CartesianRepresentation, \
                    HorizontalRepresentation, SphericalRepresentation, Rotation, Reference 
            #ExtendedCoordinateFrame, Rotation
from tests import TestCase


class CoordinatesTest(TestCase):
    '''Unit tests for the coordinates module'''

    def __init__(self, *args):
        super().__init__(*args)

        self.obstime = '2010-01-01'
        self.location = Geodetic(latitude=0., longitude=0., 
                                height=0., referece=Reference.ELLIPSOID)

    def test_geodetic(self):
        ecef = ECEF(x=6378137, y=0, z=0)
        geod = Geodetic(ecef)

        self.assertQuantity(geod.latitude , self.location.latitude , 11)
        self.assertQuantity(geod.longitude, self.location.longitude, 11)
        self.assertQuantity(geod.height   , self.location.height   , 6)

        # Check the method transformation
        geod1 = ecef.ecef_to_geodetic()
        self.assertQuantity(geod1.latitude , self.location.latitude , 11)
        self.assertQuantity(geod1.longitude, self.location.longitude, 11)
        self.assertQuantity(geod1.height   , self.location.height   , 6)

        # RK TODO: Add more test.

    def test_horizontal(self):
        '''for (angle, point) in (((90, 0), (1, 0, 0)),
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
        '''
        #RK TODO: Complete Horizontal CS and add test here.
        pass

    def test_ecef(self):
        # Check the forward transform
        ecef0 = ECEF(x=6378137, y=0, z=0)
        ecef1 = ECEF(self.location, obstime=self.obstime)
        self.assertCartesian(ecef0, ecef1, 8)

        # Check the backward transform
        ecef2 = self.location.geodetic_to_ecef()
        self.assertCartesian(ecef1, ecef2, 8)

        #RK- obstime is not yet properly used in ECEF.
        # Check the obstime handling
        '''ecef1 = itrs.transform_to(ECEF)
        self.assertEqual(ecef1.obstime, itrs.obstime)
        self.assertCartesian(ecef1, ecef0, 8)

        # Check the round trip with different obstimes
        itrs = ecef.transform_to(ITRS)
        ecef0 = itrs.transform_to(ECEF(obstime=self.obstime))
        self.assertCartesian(ecef, ecef0, 2)

        # Check the Earth location conversion
        location = ecef0.earth_location.itrs.cartesian
        self.assertCartesian(ecef, location, 2)
        '''

        '''
    #RK - ExtendedCoordinateFrame is not defined in the new coordinates.py.
          Keep this method here for sometime to see if we can use any test
          styles performed here in other frames.
    def test_extended_frame(self):
        # Test write protection
        frame0 = ExtendedCoordinateFrame(CartesianRepresentation(0, 0, 0))
        self.assertFalse(frame0._data.x.flags.writeable)
        self.assertFalse(frame0.cartesian.x.flags.writeable)
        self.assertFalse(frame0.spherical.lat.flags.writeable)

        r = frame0.cartesian.copy()
        self.assertTrue(r.x.flags.writeable)

        # Test rotations
        z4, o4 = numpy.zeros(4), numpy.ones(4)
        frame0 = ExtendedCoordinateFrame(
            SphericalRepresentation(z4 * u.deg, z4 * u.deg, o4 * u.m))
        frame1 = ExtendedCoordinateFrame(
            CartesianRepresentation(z4 * u.m, o4 * u.m, z4 * u.m))
        r = Rotation.from_euler('Z', 90 * u.deg)

        frame2 = r * frame0
        self.assertTrue(isinstance(frame2.data, SphericalRepresentation))
        self.assertFalse(frame2._data.lat.flags.writeable)
        self.assertCartesian(frame2.cartesian, frame1.cartesian)

        # Test scalar multiplication
        frame0 = ExtendedCoordinateFrame(
            SphericalRepresentation(90 * o4 * u.deg, z4 * u.deg, o4 * u.m))
        frame1 = ExtendedCoordinateFrame(
            CartesianRepresentation(z4 * u.m, 3 * o4 * u.m, z4 * u.m))

        def check(frame):
            self.assertTrue(isinstance(frame.data, SphericalRepresentation))
            self.assertFalse(frame._data.lat.flags.writeable)
            self.assertCartesian(frame.cartesian, frame1.cartesian)

        check(3 * frame0)
        check(frame0 * 3)

        # Test translation and subtraction
        frame0 = ExtendedCoordinateFrame(
            SphericalRepresentation(90 * o4 * u.deg, z4 * u.deg, o4 * u.m))
        frame1 = ExtendedCoordinateFrame(
            CartesianRepresentation(z4 * u.m, o4 * u.m, z4 * u.m))
        frame2 = ExtendedCoordinateFrame(
            CartesianRepresentation(z4 * u.m, 2 * o4 * u.m, z4 * u.m))

        def check(frame, reference, representation=SphericalRepresentation):
            self.assertTrue(isinstance(frame.data, representation))
            for component in frame._data.components:
                self.assertFalse(
                    getattr(frame._data, component).flags.writeable)
            self.assertCartesian(frame.cartesian, reference.cartesian)

        frame3 = frame0 + frame1
        check(frame3, frame2)
        check(frame3 - frame1, frame0)
        check(frame0 + frame1._data, frame2)
        check(frame3 - frame1._data, frame0)
        check(frame1._data + frame0, frame2, CartesianRepresentation)
        check(frame3._data - frame1, frame0)
        '''

    def test_ltp(self):
        ecef = ECEF(self.location, obstime=self.obstime)

        # Check the constructor & to ECEF transform
        ltp = LTP(x=0, y=0, z=0, location=self.location,
                  orientation='NWU', magnetic=False, obstime=self.obstime)
        r = ltp.ltp_to_ecef()

        self.assertEqual(r.obstime, ltp.obstime)
        self.assertCartesian(r, ecef, 6)

        # Check the from ECEF transform
        ltp = ecef.ecef_to_ltp(location=self.location, orientation='NWU',
                               magnetic=False, obstime=self.obstime)

        self.assertEqual(ltp.obstime, self.obstime)
        self.assertQuantity(ltp.x, 0, 6)
        self.assertQuantity(ltp.y, 0, 6)
        self.assertQuantity(ltp.z, 0, 6)

        # Check the Earth location conversion
        location = ECEF(ltp)
        self.assertCartesian(ecef, location, 2)

        # RK: TODO: Complete from here and below -------------------------
        # Check the affine transform
        points = ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1),
                  (0, 1, 1))
        for point in (points):
            cart = CartesianRepresentation(x=point[1], y=point[0], z=point[2])
            altaz = SkyCoord(cart, frame='altaz', location=self.location,
                             obstime=self.obstime)
            ecef0 = altaz.transform_to(ECEF(obstime=self.obstime))

            cart = CartesianRepresentation(x=point[0], y=point[1], z=point[2])
            ltp = LTP(x=cart.x, y=cart.y, z=cart.z, 
                      location=self.location, obstime=self.obstime,
                      orientation='NWU', magnetic=False)
            ecef1 = ltp.ltp_to_ecef()

            self.assertEqual(ecef0.obstime, ecef1.obstime)
            self.assertCartesian(ecef0, ecef1, 4)

        # Check the orientation
        point = (1, -1, 2)
        cart = CartesianRepresentation(x=point[0], y=point[1], z=point[2],
                                       unit=u.m)
        altaz = SkyCoord(cart, frame='altaz', location=self.location,
                         obstime=self.obstime)
        ecef0 = altaz.transform_to(ECEF(obstime=self.obstime))

        for (orientation, sign) in (('NEU', (1, 1, 1)),
                                    ('NED', (1, 1, -1)),
                                    ('SEU', (-1, 1, 1)),
                                    ('NWU', (1, -1, 1))):
            cart = CartesianRepresentation(x=sign[0] * point[0],
                y=sign[1] * point[1], z=sign[2] * point[2], unit=u.m)
            ltp = LTP(cart, location=self.location, obstime=self.obstime,
                      orientation=orientation, magnetic=False)
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
        ltp = ecef.transform_to(LTP(location=self.location, orientation='ENU',
                                    magnetic=False))

        self.assertEqual(ltp.obstime, ecef.obstime)
        self.assertQuantity(ltp.cartesian.norm(), 1 * u.one, 6)

        # Check the magnetic north case
        ltp0 = LTP(uy, location=self.location, obstime=self.obstime,
                orientation='ENU', magnetic=False)
        frame1 = LTP(location=self.location, obstime=self.obstime,
                     orientation='ENU', magnetic=True)
        ltp1 = ltp0.transform_to(frame1)
        self.assertEqual(ltp0.obstime, ltp1.obstime)

        declination = numpy.arcsin(ltp0.cartesian.cross(ltp1.cartesian).norm())
        self.assertQuantity(declination.to(u.deg), 0.10 * u.deg, 2)

        # Test the magnetic case with no obstime
        ltp1 = LTP(uy, location=self.location, orientation='ENU',
                   magnetic=True)
        self.assertIsNone(ltp1.obstime)

        # Test the invalid frame case
        with self.assertRaises(ValueError) as context:
            LTP(uy, location=self.location, orientation=('T', 'O', 'T', 'O'))
        self.assertRegex(context.exception.args[0], '^Invalid frame')

        # Test the ltp round trip with a position
        frame0 = LTP(location=self.location, orientation='ENU', magnetic=False)
        frame1 = LTP(location=self.location, obstime=self.obstime,
                     orientation='ENU', magnetic=True)
        ltp0 = LTP(x=1 * u.m, y=2 * u.m, z=3 * u.m, location=self.location,
                   orientation='ENU', magnetic=False, obstime=self.obstime)
        ltp1 = ltp0.transform_to(frame1).transform_to(frame0)
        self.assertCartesian(ltp0, ltp1, 8)

        # Test the same frame case
        ltp1 = ltp0.transform_to(frame0)
        self.assertCartesian(ltp0, ltp1, 8)
        self.assertEqual(ltp0.obstime, ltp1.obstime)

        # Test an LTP permutation
        ltp0 = LTP(x=1 * u.m, y=2 * u.m, z=3 * u.m, location=self.location,
                   orientation='ENU', magnetic=False)
        frame1 = LTP(location=self.location, orientation='NED', magnetic=False)
        ltp1 = ltp0.transform_to(frame1)
        self.assertQuantity(ltp0.x, ltp1.y, 6)
        self.assertQuantity(ltp0.y, ltp1.x, 6)
        self.assertQuantity(ltp0.z, -ltp1.z, 6)

        # Test an explicit rotation
        r = Rotation.from_euler('ZX', 90 * u.deg, 90 * u.deg)
        frame0 = LTP(location=self.location, orientation='ENU', magnetic=False,
                     rotation=r)
        frame1 = LTP(location=self.location, orientation='NUE', magnetic=False)
        self.assertArray(frame0._basis, frame1._basis)

        # Test replication with a rotation
        frame1 = frame0.rotated(r.inverse, copy=False)
        frame2 = LTP(location=self.location, orientation='ENU', magnetic=False)
        self.assertArray(frame1._basis, frame2._basis)

        # Test declination
        declination = 5 * u.deg
        frame0 = LTP(location=self.location, declination=declination,
                     orientation='ENU')
        r = Rotation.from_euler('Z', -declination)
        frame1 = LTP(location=self.location, orientation='ENU', magnetic=False,
                     rotation=r)
        self.assertArray(frame0._basis, frame1._basis)


    def test_rotation(self):
        # Test initialisation
        a0 = numpy.array((45, 30, 15)) * u.deg
        r0 = Rotation.from_euler('ZYZ', a0)
        r1 = Rotation.from_euler('ZYZ', *a0)
        self.assertArray(r0.matrix, r1.matrix)

        # Test rotation vector
        v0 = 90 * numpy.array((0, 0, 1)) * u.deg
        r0 = Rotation.from_rotvec(v0)
        v1 = r0.rotvec
        self.assertQuantity(v0, v1)

        # Test euler angles
        a0 = numpy.array((45, 30, 15)) * u.deg
        r0 = Rotation.from_euler('ZYZ', a0)
        a1 = r0.euler_angles('ZYZ')
        a2 = r0.euler_angles('ZYZ', unit='deg')
        self.assertQuantity(a0, a1)
        self.assertQuantity(a0, a2)

        # Test the application of a rotation
        r0 = Rotation.from_euler('Z', 90 * u.deg)
        r1 = r0.inverse

        v0 = numpy.array((0, 1, 0))
        v1 = numpy.array((1, 0, 0))
        v1 = r0 * v1
        self.assertArray(v0, v1)
        v1 = numpy.array((1, 0, 0))
        v1 = r1.apply(v1, inverse=True)
        self.assertArray(v0, v1)

        v0 = numpy.array((0, 1, 0)) * u.m
        v1 = numpy.array((1, 0, 0)) * u.m
        v1 = r0 * v1
        self.assertQuantity(v0, v1)
        v1 = numpy.array((1, 0, 0)) * u.m
        v1 = r1.apply(v1, inverse=True)
        self.assertQuantity(v0, v1)

        v0 = CartesianRepresentation(0, 1, 0, unit='m')
        v1 = CartesianRepresentation(1, 0, 0, unit='m')
        v1 = r0 * v1
        self.assertCartesian(v0, v1)
        v1 = CartesianRepresentation(1, 0, 0, unit='m')
        v1 = r1.apply(v1, inverse=True)
        self.assertCartesian(v0, v1)

        # Test composition of rotations
        r0 = Rotation.from_euler('z', 90 * u.deg)
        r1 = Rotation.from_euler('x', 90 * u.deg)
        r2 = Rotation.from_euler('zx', 90 * u.deg, 90 * u.deg)
        r3 = r1 * r0
        self.assertTrue(isinstance(r3, Rotation))
        self.assertArray(r3.matrix, r2.matrix)

        r3 = r1.apply(r3, inverse=True)
        self.assertTrue(isinstance(r3, Rotation))
        self.assertArray(r3.matrix, r0.matrix)

        # Test the magnitude property
        r0 = Rotation.from_euler('z', 90 * u.deg)
        self.assertQuantity(r0.magnitude, 90 * u.deg)

        # Test vectors alignment
        r0 = Rotation.from_euler('Z', 90 * u.deg)

        def check_align(v0):
            v1 = r0.apply(v0)
            r1, _ = Rotation.align_vectors(v1, v0)
            self.assertTrue(isinstance(r1, Rotation))
            self.assertArray(r0.matrix, r1.matrix)

        v0 = numpy.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        check_align(v0)

        v0 = numpy.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))) * u.m
        check_align(v0)

        v0 = CartesianRepresentation((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 0),
                                     unit='m')
        check_align(v0)

        # Test the identity generator
        e3 = numpy.eye(3)

        def check_identity(r):
            self.assertTrue(isinstance(r, Rotation))
            self.assertArray(r.matrix, e3)

        r0 = Rotation.identity()
        check_identity(r0)

        rs = Rotation.identity(3)
        for r0 in rs:
            check_identity(r0)

        # Test the random generator
        def check_random(r):
            self.assertTrue(isinstance(r, Rotation))

        r0 = Rotation.random()
        check_random(r0)

        rs = Rotation.random(3)
        for r0 in rs:
            check_random(r0)


if __name__ == '__main__':
    unittest.main()
