# -*- coding: utf-8 -*-
"""
Unit tests for the grand.radio.detector module

Usage: python -m tests.radio.test_detector

"""

import unittest
import sys

import astropy.units as u
from grand.radio.detector import Detector
import numpy as np


class DetectorTest(unittest.TestCase):
    """Unit tests for the detector module

    TODO: add tests for missing modules

    """

    def assertQuantity(self, x, y):
        """Check that two astropy.Quantities are consistent"""
        self.assertEqual((x / x.unit).value, (y / x.unit).value)

    def test_type(self):
        type_array = np.array([ "LDPA", "butterfly", "Horizon" ])

        det = Detector()
        det.type = type_array[:].tolist()
        types = det.type

        self.assertEqual(type_array[1], types[1]) #shall be equal
        self.assertEqual(len(type_array), len(types))


    def test_position(self):
        pos_array = np.array([ [100, 100, 3000], [200, 300, 2500], [10, 500, -100] ])

        det = Detector()
        det.position = pos_array[:].tolist() # XXX should be an astropy.Quantity
        positions = det.position

        self.assertQuantity(pos_array[1,1] * u.m, positions[1,1]) #shall be equal
        self.assertEqual(len(pos_array), len(positions))


    def test_ID(self):
        ID_array = np.array([1,43,600,"ant1", 99])

        det = Detector()
        det.ID = ID_array[:].tolist()
        ID = det.ID

        self.assertEqual(ID_array[1], ID[1]) #shall be equal
        self.assertEqual(len(ID_array), len(ID))


    def test_location(self):
        det = Detector()

        det.location = "Fancy place"

        self.assertEqual("Fancy place", det.location) #shall be equal


    def test_origin(self):
        det = Detector()

        origin_default = ( 0 * u.m, 0 * u.m, 0 * u.m)
        det.origin = origin_default

        self.assertQuantity(origin_default[0], det.origin[0]) #shall be equal


if __name__ == "__main__":
    unittest.main()
