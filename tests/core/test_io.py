"""
Unit tests for the grand.hdf5 module
"""

from pathlib import Path
import unittest

# from astropy.coordinates import Angle, CartesianRepresentation,                \
#                                UnitSphericalRep.resentation
# import astropy.units as u
import numpy

import grand.dataio.hdf5 as io
from grand import (
    Geodetic,
    ECEF,
    LTP,
    Rotation,
    CartesianRepresentation,
    SphericalRepresentation,
)
from tests import TestCase


class IoTest(TestCase):
    """Unit tests for grand.hdf5 class"""

    path = Path("dataio.hdf5")

    def tearDown(self):
        self.path.unlink()

    """def test_readwrite(self):
        r0 = CartesianRepresentation(1 * u.m, 2 * u.m, 3 * u.m)
        u0 = UnitSphericalRepresentation(90 * u.deg, -90 * u.deg)
        c = numpy.array((1, 2)) * u.m
        r1 = CartesianRepresentation(c, c, c)
        c = Angle(90, unit=u.deg)
        u1 = UnitSphericalRepresentation(c, c)
        loc = ECEF(45 * u.deg, 6 * u.deg, 0 * u.m,
                   representation_type='geodetic')

        elements = {
            'primary'    : 'pà',
            'bytes'      : b'0100011',
            'id'         : 1,
            'energy0'    : 1.0,
            'energy1'    : 1 * u.eV,
            'data'       : numpy.array(((1, 2, 3), (4, 5, 6))),
            'position0'  : r0.xyz,
            'position1'  : [r0.x, r0.y, r0.z],
            'position2'  : r0,
            'position3'  : r1,
            'direction0' : u0,
            'direction1' : u1,
            'frame0'     : ECEF(obstime='2010-01-01'),
            'frame1'     : LTP(location=loc, obstime='2010-01-01',
                               magnetic=True,
                               rotation=Rotation.from_euler('z', 90 * u.deg))
        }

        with dataio.open(self.path, 'w') as root:
            for k, v in elements.items():
                root.write(k, v)

        with dataio.open(self.path) as root:
            for name, element in root.elements:
                a = elements[name]

                if hasattr(a, 'shape'):
                    self.assertEquals(a.shape, element.shape)

                if isinstance(a, u.Quantity):
                    self.assertQuantity(a, element)
                elif isinstance(a, CartesianRepresentation):
                    self.assertCartesian(a, element)
                elif isinstance(a, UnitSphericalRepresentation):
                    self.assertUnitSpherical(a, element)
                elif isinstance(a, numpy.ndarray):
                    self.assertEquals(a.shape, element.shape)
                    self.assertArray(a, element)
                elif isinstance(a, ECEF):
                    self.assertEquals(a.obstime.jd, element.obstime.jd)
                elif isinstance(a, LTP):
                    self.assertEquals(a.obstime.jd, element.obstime.jd)
                    self.assertCartesian(a.location.itrs.cartesian,
                                         element.location.itrs.cartesian, 8)
                    self.assertEquals(a.orientation, element.orientation)
                    self.assertEquals(a.magnetic, element.magnetic)
                    self.assertArray(a.rotation.matrix, element.rotation.matrix)
                    self.assertArray(a._basis, element._basis)
                    self.assertCartesian(a._origin, element._origin, 8)
                else:
                    self.assertEquals(a, element)
    """
    # RK
    def test_readwrite(self):
        r0 = CartesianRepresentation(x=1, y=2, z=3)
        u0 = SphericalRepresentation(theta=90, phi=-90, r=1)
        c = numpy.array((1, 2))
        r1 = CartesianRepresentation(x=c, y=c, z=c)
        c = 90
        u1 = SphericalRepresentation(theta=c, phi=c, r=1)
        loc = Geodetic(latitude=45, longitude=6, height=0)

        elements = {
            "primary": "pà",
            "bytes": b"0100011",
            "id": 1,
            "energy0": 1.0,
            "energy1": 1,
            "data": numpy.array(((1, 2, 3), (4, 5, 6))),
            "position0": r0,
            "position1": [r0.x, r0.y, r0.z],
            "position2": r0,
            "position3": r1,
            "direction0": u0,
            "direction1": u1,
            "frame0": ECEF(x=0, y=0, z=0, obstime="2010-01-01"),
            "frame1": LTP(
                x=0,
                y=0,
                z=0,
                location=loc,
                obstime="2010-01-01",
                magnetic=True,
                orientation="NWU",
            )
            # rotation=Rotation.from_euler('z', 90 * u.deg)) #RK. TODO.
        }

        with io.open(self.path, "w") as root:
            for k, v in elements.items():
                root.write(k, v)

        with io.open(self.path) as root:
            for name, element in root.elements:
                a = elements[name]
                print(a, type(a))
                if hasattr(a, "shape"):
                    self.assertEqual(a.shape, element.shape)

                # RK: TODO: ECEF, LTP is not detected by isinstance, instead
                #           they are seen as CartesianRepresentation objects.
                if isinstance(a, CartesianRepresentation):
                    self.assertCartesian(a, element)
                elif isinstance(a, SphericalRepresentation):
                    self.assertSpherical(a, element)
                elif isinstance(a, numpy.ndarray):
                    self.assertEqual(a.shape, element.shape)
                    self.assertArray(a, element)
                elif isinstance(a, ECEF):
                    self.assertEqual(a.obstime, element.obstime)
                elif isinstance(a, LTP):
                    self.assertEqual(a.obstime, element.obstime)
                    self.assertCartesian(a.location, element.location, 8)
                    self.assertEqual(a.orientation, element.orientation)
                    self.assertEqual(a.magnetic, element.magnetic)
                    # self.assertArray(a.rotation.matrix, element.rotation.matrix)
                    self.assertArray(a.basis, element.basis)
                else:
                    self.assertEqual(a, element)


if __name__ == "__main__":
    unittest.main()
