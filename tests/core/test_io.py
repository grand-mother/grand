"""
Unit tests for the grand.io module
"""

from pathlib import Path
import unittest

from astropy.coordinates import CartesianRepresentation,                       \
                                UnitSphericalRepresentation
import astropy.units as u
import numpy

import grand.io as io
from tests import TestCase


class IoTest(TestCase):
    """Unit tests for grand.io class"""

    path = Path("io.hdf5")

    def tearDown(self):
        self.path.unlink()

    def test_readwrite(self):
        r0 = CartesianRepresentation(1 * u.m, 2 * u.m, 3 * u.m)
        u0 = UnitSphericalRepresentation(90 * u.deg, -90 * u.deg)
        c = numpy.array((1, 2)) * u.m
        r1 = CartesianRepresentation(c, c, c)
        c = numpy.array((90, -90)) * u.deg
        u1 = UnitSphericalRepresentation(c, c)

        elements = {
            "primary"    : "pà",
            "bytes"      : b"0100011",
            "id"         : 1,
            "energy0"    : 1.0,
            "energy1"    : 1 * u.eV,
            "data"       : numpy.array(((1, 2, 3), (4, 5, 6))),
            "position0"  : r0.xyz,
            "position1"  : [r0.x, r0.y, r0.z],
            "position2"  : r0,
            "position3"  : r1,
            "direction0" : u0,
            "direction1" : u1
        }

        with io.open(self.path, "w") as root:
            for k, v in elements.items():
                root.write(k, v)

        with io.open(self.path) as root:
            for name, element in root.elements:
                a = elements[name]

                if hasattr(a, "shape"):
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
                else:
                    self.assertEquals(a, element)

if __name__ == "__main__":
    unittest.main()
