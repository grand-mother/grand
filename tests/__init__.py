"""
Unit tests for the GRAND package
"""

import argparse
import doctest
import os
import unittest
import sys

from pathlib import Path

__all__ = ["main"]


class TestCase(unittest.TestCase):
    def assertArray(self, a, b, tol=9):
        """Check that two numpy.ndarray are consistent"""
        if len(a.shape) > 1:
            a, b = a.flatten(), b.flatten()
        for i, ai in enumerate(a): self.assertAlmostEqual(ai, b[i], tol)

    def assertQuantity(self, a, b, tol=9):
        """Check that two astropy.Quantities are consistent"""
        n = a.size
        self.assertEqual(n, b.size)
        b = b.to_value(a.unit)
        a = a.value
        if n > 1:
            if len(a.shape) > 1:
                a, b = a.flatten(), b.flatten()
            for i, ai in enumerate(a): self.assertAlmostEqual(ai, b[i], tol)
        else:
            self.assertAlmostEqual(a, b, tol)

    def assertCartesian(self, a, b, tol=9):
        """Check that two CartesianRepresentations are consistent"""
        self.assertQuantity(a.x, b.x, tol)
        self.assertQuantity(a.y, b.y, tol)
        self.assertQuantity(a.z, b.z, tol)

    def assertUnitSpherical(self, a, b, tol=9):
        """Check that two UnitSphericalRepresentations are consistent"""
        self.assertQuantity(a.lon, b.lon, tol)
        self.assertQuantity(a.lat, b.lat, tol)


def main():
    """Run a local test suite
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbosity", help="set output verbosity", type=int, default=2)
    parser.add_argument(
        "--doc", help="run only doc tests", action="store_true", default=False)
    parser.add_argument(
        "--unit", help="run only unit tests", action="store_true",
        default=False)
    options = parser.parse_args()

    dirname = Path(__file__).parent
    localdir = Path(sys.argv[0]).parent
    exit_status = 0

    if not options.doc:
        # Look for unit tests
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(localdir, pattern="test_*.py")

        # Run the unit tests
        if test_suite.countTestCases():
            print("running unit tests...")
            if options.verbosity > 1:
                print(70 * "=")

            runner = unittest.TextTestRunner(verbosity=options.verbosity)
            exit_status = not runner.run(test_suite).wasSuccessful()
            if exit_status:
                sys.exit(exit_status)

    if not options.unit:
        # Look for doc tests
        from astropy.coordinates import CartesianRepresentation
        import astropy.units as u
        from grand import geomagnet, store, topography, ECEF,                  \
                          GeodeticRepresentation, HorizontalRepresentation, LTP
        import grand.io as io
        globs = { "geomagnet": geomagnet, "io": io, "store": store,
                  "topography": topography,
                  "CartesianRepresentation": CartesianRepresentation,
                  "ECEF": ECEF, "LTP": LTP,
                  "GeodeticRepresentation": GeodeticRepresentation,
                  "HorizontalRepresentation": HorizontalRepresentation, "u": u}

        test_suite = unittest.TestSuite()
        top = dirname.parent
        relative = localdir.relative_to(dirname)
        for root, _, filenames in os.walk(top / "docs" / relative):
            for filename in filenames:
                if filename[-4:] != '.rst':
                    continue
                path = os.path.join(root, filename)
                test_suite.addTest(doctest.DocFileSuite(path,
                        module_relative=False, globs=globs))

        # Run the doc tests
        if test_suite.countTestCases():
            print()
            print("running doc tests...")
            if options.verbosity > 1:
                print(70 * "=")

            runner = unittest.TextTestRunner(verbosity=options.verbosity)
            exit_status = not runner.run(test_suite).wasSuccessful()
            try:
                os.remove("data.hdf5")
            except FileNotFoundError:
                pass

    sys.exit(exit_status)
