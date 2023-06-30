"""
Unit tests for the grand_tools.coordinates module

RK Todo: Complete Horizontal coordinate system and 
         implement Rotation of LTP frame.
"""
import unittest
from tests import TestCase

import numpy
import datetime
from grand import (
    ECEF,
    LTP,
    Geodetic,
    GeodeticRepresentation,
    CartesianRepresentation,
    SphericalRepresentation,
    HorizontalRepresentation,
    HorizontalVector,
    Horizontal,
    Reference,
    Coordinates,
    Rotation,
    GRANDCS,
)
import copy as _copy


def copy(obj, deep=False):
    import inspect

    if deep:
        new = _copy.deepcopy(obj)
        for var in vars(obj):
            attr = getattr(obj, var)
            setattr(new, var, _copy.deepcopy(attr))
    else:
        new = _copy.copy(obj)
        for var in vars(obj):
            attr = getattr(obj, var)
            setattr(new, var, _copy.copy(attr))

    return new


class CoordinatesTest(TestCase):
    """Unit tests for the coordinates module"""

    def __init__(self, *args):
        super().__init__(*args)

        self.obstime = "2020-01-01"
        self.location = Geodetic(latitude=0.0, longitude=0.0, height=0.0, reference="ELLIPSOID")

    # RK
    def test_coordinates(self):
        ar1 = Coordinates(1)
        ar2 = Coordinates(5)
        self.assertEqual(ar1.shape[0], 3)
        self.assertEqual(ar1.shape[1], 1)
        self.assertEqual(ar2.shape[0], 3)
        self.assertEqual(ar2.shape[1], 5)
        # make sure raise TypeError if input is not integer.
        with self.assertRaises(TypeError) as context:
            Coordinates(5.0)

    # RK
    def test_cartesianrepresentation(self):
        cart1 = CartesianRepresentation(x=1, y=1, z=0)
        norm1 = cart1.norm()
        sphr1 = SphericalRepresentation(cart1)
        sphr2 = cart1.cartesian_to_spherical()
        horz2 = cart1.cartesian_to_horizontal()
        self.assertEqual(norm1, numpy.sqrt(cart1.x ** 2 + cart1.y ** 2 + cart1.z ** 2))
        self.assertEqual(sphr1.theta, 90)
        self.assertEqual(sphr1.phi, 45)
        self.assertEqual(sphr1.r, numpy.sqrt(2))
        self.assertEqual(sphr2.theta, 90)
        self.assertEqual(sphr2.phi, 45)
        self.assertEqual(sphr2.r, numpy.sqrt(2))
        self.assertEqual(horz2.azimuth, 45)
        self.assertEqual(horz2.elevation, 0)
        # check getter. Use assertQuantity or assertEqual
        self.assertQuantity(cart1.x, cart1[0], 6)
        self.assertQuantity(cart1.y, cart1[1], 6)
        self.assertQuantity(cart1.z, cart1[2], 6)
        with self.assertRaises(TypeError) as context:
            CartesianRepresentation(x="one", y=1, z=0)

    # RK
    def test_sphericalrepresentation(self):
        sphr1 = SphericalRepresentation(theta=90, phi=45, r=numpy.sqrt(2))
        cart1 = CartesianRepresentation(sphr1)
        cart2 = sphr1.spherical_to_cartesian()
        horz2 = sphr1.spherical_to_horizontal()
        self.assertEqual(sphr1.theta[0], 90)
        self.assertEqual(sphr1.phi[0], 45)
        self.assertEqual(sphr1.r[0], numpy.sqrt(2))
        self.assertQuantity(cart1.x[0], 1, 6)
        self.assertQuantity(cart1.y[0], 1, 6)
        self.assertQuantity(cart1.z[0], 0, 6)
        self.assertQuantity(cart2.x[0], 1, 6)
        self.assertQuantity(cart2.y[0], 1, 6)
        self.assertQuantity(cart2.z[0], 0, 6)
        self.assertQuantity(horz2.azimuth[0], 45)
        self.assertQuantity(horz2.elevation[0], 0)
        self.assertQuantity(horz2.norm[0], numpy.sqrt(2))
        with self.assertRaises(TypeError) as context:
            SphericalRepresentation(theta="one", phi=45, r=1.41)

    # RK
    def test_geodeticrepresentation(self):
        geod = GeodeticRepresentation(latitude=0, longitude=0, height=0)
        # test getter
        self.assertQuantity(geod.latitude[0], 0, 6)
        self.assertQuantity(geod.longitude[0], 0, 6)
        self.assertQuantity(geod.height[0], 0, 6)
        # test setter
        geod.latitude += 1
        geod.longitude += 1
        geod.height += 1
        self.assertQuantity(geod.latitude[0], 1, 6)
        self.assertQuantity(geod.longitude[0], 1, 6)
        self.assertQuantity(geod.height[0], 1, 6)
        # test TypeError
        with self.assertRaises(TypeError) as context:
            GeodeticRepresentation(latitude="zero", longitude=0, height=0)

    # RK. Horizontal CS is not complete. Rework on this.
    def test_horizontalrepresentation(self):
        horz = HorizontalRepresentation(azimuth=45, elevation=0, norm=numpy.sqrt(2))
        cart = horz.horizontal_to_cartesian()
        sphr = horz.horizontal_to_spherical()
        # test getter
        self.assertQuantity(horz.azimuth[0], 45, 6)
        self.assertQuantity(horz.elevation[0], 0, 6)
        self.assertQuantity(horz.norm[0], numpy.sqrt(2), 6)
        # test setter
        horz.azimuth += 1
        horz.elevation += 1
        horz.norm += 1
        self.assertQuantity(horz.azimuth[0], 46, 6)
        self.assertQuantity(horz.elevation[0], 1, 6)
        self.assertQuantity(horz.norm[0], numpy.sqrt(2) + 1, 6)
        # test TypeError
        with self.assertRaises(TypeError) as context:
            GeodeticRepresentation(azimuth="zero", elevation=0, norm=0)

        # test transformation
        self.assertAlmostEqual(cart.x[0], 1, 6)
        self.assertQuantity(cart.y[0], 1, 6)
        self.assertQuantity(cart.z[0], 0, 6)
        self.assertQuantity(sphr.theta[0], 90)
        self.assertQuantity(sphr.phi[0], 45)
        self.assertQuantity(sphr.r[0], numpy.sqrt(2))

    def test_geodetic(self):
        ecef = ECEF(x=6378137, y=0, z=0)
        geod = Geodetic(ecef, reference="ELLIPSOID")
        ecef1 = geod.geodetic_to_ecef()
        geod1 = ecef.ecef_to_geodetic(reference="ELLIPSOID")
        horz = geod.geodetic_to_horizontal()
        grnd = geod.geodetic_to_grandcs()
        ltpf = LTP(location=self.location, orientation="NWU")
        ltp = geod.geodetic_to_ltp(ltpf)

        # Check the method transformation
        self.assertQuantity(geod, self.location, 6)
        self.assertQuantity(geod1, self.location, 6)
        self.assertQuantity(ecef, ecef1, 6)
        self.assertCartesian(ecef, ecef1, 6)

        # RK TODO: Add more test.
        self.assertEqual(geod.reference, "ELLIPSOID")
        self.assertQuantity(ltp.location, ecef)
        # check input value is either number or array of numbers.
        with self.assertRaises(TypeError) as context:
            Geodetic(azimuth="zero", elevation=0, norm=0)
        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            Geodetic(numpy.ones(10))

    def test_horizontal(self):
        """for (angle, point) in (((90, 0), (1, 0, 0)),
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
        """
        # RK TODO: Complete Horizontal CS and add test here.
        pass

    def test_ecef(self):
        # Check the forward transform
        ecef0 = ECEF(x=6378137, y=0, z=0)
        ecef1 = ECEF(self.location)
        ecef2 = self.location.geodetic_to_ecef()
        geod = ecef0.ecef_to_geodetic(reference="ELLIPSOID")
        self.assertCartesian(ecef0, ecef1, 8)
        self.assertCartesian(ecef1, ecef2, 8)
        self.assertQuantity(geod, self.location)
        self.assertQuantity(geod.latitude[0], 0)
        self.assertEqual(ecef0.obstime, "2020-01-01")
        # check input value is either number or array of numbers.
        with self.assertRaises(TypeError) as context:
            ECEF(x="zero", y=0, z=0)
        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            ECEF(numpy.ones(10))

        # Check the backward transform

        # RK- obstime is not yet properly used in ECEF.
        # Check the obstime handling
        """ecef1 = itrs.transform_to(ECEF)
        self.assertEqual(ecef1.obstime, itrs.obstime)
        self.assertCartesian(ecef1, ecef0, 8)

        # Check the round trip with different obstimes
        itrs = ecef.transform_to(ITRS)
        ecef0 = itrs.transform_to(ECEF(obstime=self.obstime))
        self.assertCartesian(ecef, ecef0, 2)

        # Check the Earth location conversion
        location = ecef0.earth_location.itrs.cartesian
        self.assertCartesian(ecef, location, 2)
        """

        """
    #RK - ExtendedCoordinateFrame is not defined in the new coordinates.py.
          Keep this method here for sometime to see if we can use any test
          styles performed here to test other frames.
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
        """

    def test_ltp(self):
        loc = self.location
        location_ref0 = self.location.reference
        loc_ref0 = loc.reference

        ecef = ECEF(self.location)
        # Check the constructor & to ECEF transform
        ltp = LTP(
            x=0,
            y=0,
            z=0,
            location=self.location,
            orientation="NWU",
            magnetic=False,
            obstime=self.obstime,
        )
        r = ltp.ltp_to_ecef()

        self.assertEqual(r.obstime, ltp.obstime)
        self.assertCartesian(r, ecef, 6)

        # Check the from ECEF transform
        ltp_frame = LTP(
            location=self.location,
            orientation="ENU",
            magnetic=False,
            obstime=self.obstime,
        )
        # ltp = ecef.ecef_to_ltp(ltp_frame)
        ltp = LTP(ecef, frame=ltp_frame)
        # RK Note:
        # self.location.reference can suddenly changes from ELLIPSOID to GEOID
        # reference changes if it is stored as class attribute instead of instance
        # attribute. Always check reference consitency for a Geodetic coordinate.
        self.assertEqual(self.location.reference, "ELLIPSOID")
        self.assertEqual(loc.reference, "ELLIPSOID")

        # attributes can change if they are class attribute instead of instance attribute.
        # always check attributes consitency for a LTP coordinate.
        self.assertEqual(ltp.obstime, self.obstime)
        self.assertQuantity(ltp.x, numpy.zeros(1), 6)
        self.assertQuantity(ltp.y, numpy.zeros(1), 6)
        self.assertQuantity(ltp.z, numpy.zeros(1), 6)

        # Check the Earth location conversion
        loc = ECEF(ltp)
        self.assertCartesian(ecef, loc, 2)
        # Check the affine transform.
        points = ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))
        for point in points:

            cart = CartesianRepresentation(x=point[0], y=point[1], z=point[2])
            ltp = LTP(
                x=cart.x,
                y=cart.y,
                z=cart.z,
                location=self.location,
                obstime=self.obstime,
                orientation="NWU",
                magnetic=False,
            )
            ecef0 = ECEF(ltp)
            ecef1 = ltp.ltp_to_ecef()

            self.assertEqual(ecef0.obstime, ecef1.obstime)
            self.assertCartesian(ecef0, ecef1, 4)

        # Check the orientation
        point = (1, -1, 2)

        ltp = LTP(
            x=point[0],
            y=point[1],
            z=point[2],
            location=self.location,
            obstime=self.obstime,
            orientation="NEU",
        )
        ecef0 = ltp.ltp_to_ecef()

        for (orientation, sign) in (
            ("NEU", (1, 1, 1)),
            ("NED", (1, 1, -1)),
            ("SEU", (-1, 1, 1)),
            ("NWU", (1, -1, 1)),
        ):
            ltp = LTP(
                x=sign[0] * point[0],
                y=sign[1] * point[1],
                z=sign[2] * point[2],
                location=self.location,
                obstime=self.obstime,
                orientation=orientation,
                magnetic=False,
            )
            ecef1 = ECEF(ltp)
            self.assertCartesian(ecef0, ecef1, 4)

        # Check the unit vector case.
        # RK TODO: Develop HorizontalRepresentation, it is not complete and might have some error.
        uy = Horizontal(azimuth=0, elevation=0, location=self.location)
        ecef = uy.horizontal_to_ecef()
        ltp = LTP(ecef, location=self.location, obstime=self.obstime, orientation="ENU")
        self.assertQuantity(ltp.x, numpy.zeros(1), 9)
        self.assertQuantity(ltp.y, numpy.ones(1), 9)
        self.assertQuantity(ltp.z, numpy.zeros(1), 9)

        # r = ECEF(ltp)         #RK. Use this or the next.
        r = ltp.ltp_to_ecef()  # RK

        self.assertEqual(r.obstime, ltp.obstime)
        self.assertQuantity(r.norm(), 6378137.0, 6)  # RK

        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            LTP(numpy.ones(10), frame=ltp)
        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            LTP(location=numpy.ones(10), orientation="NWU")

        # RK: Horizontal coordinate system is incomplete and need more work.
        """
        ecef = ECEF(uy, obstime=self.obstime)
        #ltp = LTP(ecef, location=self.location, orientation='ENU', magnetic=False) #RK, or
        ltp = ecef.ecef_to_ltp(ltp_frame)                                           #RK
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
        """

        # Test the ltp round trip with a position
        frame0 = LTP(location=self.location, orientation="ENU", magnetic=False)
        frame1 = LTP(
            location=self.location,
            obstime=self.obstime,
            orientation="ENU",
            magnetic=True,
        )
        ltp0 = LTP(
            x=1,
            y=2,
            z=3,
            location=self.location,
            orientation="ENU",
            magnetic=False,
            obstime=self.obstime,
        )  # RK
        ltp1_ = ltp0.ltp_to_ltp(frame1)  # RK
        ltp1 = ltp1_.ltp_to_ltp(frame0)  # RK

        self.assertCartesian(ltp0, ltp1, 8)

        # Test the same frame case
        ltp1 = ltp0.ltp_to_ltp(frame0)  # RK
        self.assertCartesian(ltp0, ltp1, 8)
        self.assertEqual(ltp0.obstime, ltp1.obstime)

        # Test an LTP permutation
        ltp0 = LTP(x=1, y=2, z=3, location=self.location, orientation="ENU", magnetic=False)  # RK
        frame1 = LTP(location=self.location, orientation="NED", magnetic=False)
        ltp1 = ltp0.ltp_to_ltp(frame1)  # RK
        self.assertQuantity(ltp0.x, ltp1.y, 6)
        self.assertQuantity(ltp0.y, ltp1.x, 6)
        self.assertQuantity(ltp0.z, -ltp1.z, 6)

        """
        # Test an explicit rotation
        # RK: Rework on Rotation
        #r = Rotation.from_euler('ZX', 90 * u.deg, 90 * u.deg)
        r = Rotation.from_euler('ZX', 90 , 90) #RK
        frame0 = LTP(location=self.location, orientation='ENU', magnetic=False,
                     rotation=r)
        frame1 = LTP(location=self.location, orientation='NUE', magnetic=False)
        #self.assertArray(frame0._basis, frame1._basis)
        self.assertArray(frame0.basis, frame1.basis) #RK

        # Test replication with a rotation
        frame1 = frame0.rotated(r.inverse, copy=False)
        frame2 = LTP(location=self.location, orientation='ENU', magnetic=False)
        #self.assertArray(frame1._basis, frame2._basis)
        self.assertArray(frame1.basis, frame2.basis) #RK

        # Test declination
        declination = 5 #* u.deg
        frame0 = LTP(location=self.location, declination=declination,
                     orientation='ENU')
        r = Rotation.from_euler('Z', -declination)
        frame1 = LTP(location=self.location, orientation='ENU', magnetic=False,
                     rotation=r)
        #self.assertArray(frame0._basis, frame1._basis)
        self.assertArray(frame0.basis, frame1.basis)
        

    def test_rotation(self):
        # Test initialisation
        a0 = numpy.array((45, 30, 15)) #* u.deg
        r0 = Rotation.from_euler('ZYZ', a0)
        r1 = Rotation.from_euler('ZYZ', *a0)
        self.assertArray(r0.matrix, r1.matrix)

        # Test rotation vector
        v0 = 90 * numpy.array((0, 0, 1)) #* u.deg
        r0 = Rotation.from_rotvec(v0)
        v1 = r0.rotvec
        self.assertQuantity(v0, v1)

        # Test euler angles
        a0 = numpy.array((45, 30, 15)) #* u.deg
        r0 = Rotation.from_euler('ZYZ', a0)
        a1 = r0.euler_angles('ZYZ')
        a2 = r0.euler_angles('ZYZ', unit='deg')
        self.assertQuantity(a0, a1)
        self.assertQuantity(a0, a2)

        # Test the application of a rotation
        r0 = Rotation.from_euler('Z', 90) #* u.deg)
        r1 = r0.inverse

        v0 = numpy.array((0, 1, 0))
        v1 = numpy.array((1, 0, 0))
        v1 = r0 * v1
        self.assertArray(v0, v1)
        v1 = numpy.array((1, 0, 0))
        v1 = r1.apply(v1, inverse=True)
        self.assertArray(v0, v1)

        v0 = numpy.array((0, 1, 0)) #* u.m
        v1 = numpy.array((1, 0, 0)) #* u.m
        v1 = r0 * v1
        self.assertQuantity(v0, v1)
        v1 = numpy.array((1, 0, 0)) #* u.m
        v1 = r1.apply(v1, inverse=True)
        self.assertQuantity(v0, v1)

        #v0 = CartesianRepresentation(0, 1, 0, unit='m')
        #v1 = CartesianRepresentation(1, 0, 0, unit='m')
        v0 = CartesianRepresentation(x=0, y=1, z=0)
        v1 = CartesianRepresentation(x=1, y=0, z=0)
        v1 = r0 * v1
        self.assertCartesian(v0, v1)
        #v1 = CartesianRepresentation(1, 0, 0, unit='m')
        v1 = CartesianRepresentation(x=1, y=0, z=0)
        v1 = r1.apply(v1, inverse=True)
        self.assertCartesian(v0, v1)

        # Test composition of rotations
        r0 = Rotation.from_euler('z', 90)# * u.deg)
        r1 = Rotation.from_euler('x', 90)# * u.deg)
        #r2 = Rotation.from_euler('zx', 90 * u.deg, 90 * u.deg)
        r2 = Rotation.from_euler('zx', 90, 90)
        r3 = r1 * r0
        self.assertTrue(isinstance(r3, Rotation))
        self.assertArray(r3.matrix, r2.matrix)

        r3 = r1.apply(r3, inverse=True)
        self.assertTrue(isinstance(r3, Rotation))
        self.assertArray(r3.matrix, r0.matrix)

        # Test the magnitude property
        r0 = Rotation.from_euler('z', 90)# * u.deg)
        self.assertQuantity(r0.magnitude, 90)# * u.deg)

        # Test vectors alignment
        r0 = Rotation.from_euler('Z', 90)# * u.deg)

        def check_align(v0):
            v1 = r0.apply(v0)
            r1, _ = Rotation.align_vectors(v1, v0)
            self.assertTrue(isinstance(r1, Rotation))
            self.assertArray(r0.matrix, r1.matrix)

        v0 = numpy.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        check_align(v0)

        v0 = numpy.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))) #* u.m
        check_align(v0)

        v0 = CartesianRepresentation(x=(1, 0, 0, 1), y=(0, 1, 0, 1), z=(0, 0, 1, 0))#,unit='m')
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
        """

    def test_grandcs(self):
        # RK. Add more tests.
        grnd = GRANDCS(x=0, y=0, z=0, location=self.location)
        ecef = grnd.grandcs_to_ecef()
        geod = grnd.grandcs_to_geodetic(reference="ELLIPSOID")

        self.assertQuantity(grnd.location, ECEF(self.location))
        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            GRANDCS(numpy.ones(10))
        # check input argument is of knonw coordinate system.
        with self.assertRaises(TypeError) as context:
            GRANDCS(x="one", y=0, z=0)


if __name__ == "__main__":
    unittest.main()
