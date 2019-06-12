# -*- coding: utf-8 -*-
"""
Unit tests for the grand.libs.gull module
"""

import os
import unittest

from grand.libs import gull


class GullTest(unittest.TestCase):
    """Unit tests for the gull sub-package"""

    def test_init(self):
        self.assertEqual(gull.LIBNAME, "libgull.so")
        self.assertNotEqual(gull.LIBHASH, None)


    def test_install(self):
        self.assertTrue(os.path.exists(gull.LIBPATH))


    def test_load(self):
        self.assertNotEqual(gull._lib, None)


    def test_snapshot(self):
        snapshot = gull.Snapshot()
        self.assertNotEqual(snapshot._snapshot, None)
        self.assertEqual(snapshot.model, "IGRF12")
        d = snapshot.date
        self.assertEqual(d.year, 2019)
        self.assertEqual(d.month, 1)
        self.assertEqual(d.day, 1)
        self.assertEqual(snapshot.order, 13)
        self.assertEqual(snapshot.altitude[0], -1E+03)
        self.assertEqual(snapshot.altitude[1], 600E+03)
        del snapshot

        snapshot = gull.Snapshot("WMM2015", "2018-06-04")
        self.assertNotEqual(snapshot._snapshot, None)
        self.assertEqual(snapshot.model, "WMM2015")
        d = snapshot.date
        self.assertEqual(d.year, 2018)
        self.assertEqual(d.month, 6)
        self.assertEqual(d.day, 4)

        # Magnetic field according to
        # http://geomag.nrcan.gc.ca/calc/mfcal-en.php
        ref = (0, 2.2983E-05, -4.0852E-05)

        m = snapshot(45., 3.)
        tol = 6
        self.assertAlmostEqual(m[0], ref[0], tol)
        self.assertAlmostEqual(m[1], ref[1], tol)
        self.assertAlmostEqual(m[2], ref[2], tol)

        n = 10
        m = snapshot(n * (45.,), n * (3.,))
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
