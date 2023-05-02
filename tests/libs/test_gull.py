"""
Unit tests for the grand.libs.gull module
"""

import os
import unittest

from grand.geo import gull
from tests import TestCase


class GullTest(TestCase):
    """Unit tests for the gull sub-package"""

    def test_snapshot(self):
        snapshot = gull.Snapshot()
        self.assertNotEqual(snapshot._snapshot, None)
        self.assertEqual(snapshot.model, "IGRF13")
        d = snapshot.date
        self.assertEqual(d.year, 2020)
        self.assertEqual(d.month, 1)
        self.assertEqual(d.day, 1)
        self.assertEqual(snapshot.order, 13)
        self.assertEqual(snapshot.altitude[0], -1e03)
        self.assertEqual(snapshot.altitude[1], 600e03)
        del snapshot

        snapshot = gull.Snapshot("WMM2020", "2020-03-23")
        self.assertNotEqual(snapshot._snapshot, None)
        self.assertEqual(snapshot.model, "WMM2020")
        d = snapshot.date
        self.assertEqual(d.year, 2020)
        self.assertEqual(d.month, 3)
        self.assertEqual(d.day, 23)

        # Magnetic field according to
        # https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
        ref = (566e-09, 22999e-09, -41003e-09)

        m = snapshot(45.0, 3.0)
        tol = 6
        self.assertAlmostEqual(m[0], ref[0], tol)
        self.assertAlmostEqual(m[1], ref[1], tol)
        self.assertAlmostEqual(m[2], ref[2], tol)

        n = 10
        m = snapshot(n * (45.0,), n * (3.0,))
        self.assertEqual(m.shape[0], n)
        self.assertEqual(m.shape[1], 3)
        for i in range(n):
            self.assertAlmostEqual(m[i, 0], ref[0], tol)
            self.assertAlmostEqual(m[i, 1], ref[1], tol)
            self.assertAlmostEqual(m[i, 2], ref[2], tol)

    def test_snapshot_error(self):
        with self.assertRaises(gull.LibraryError) as context:
            snapshot = gull.Snapshot("Unknown")


if __name__ == "__main__":
    unittest.main()
