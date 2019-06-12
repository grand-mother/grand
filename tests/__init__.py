# -*- coding: utf-8 -*-
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
        from grand import geomagnet, store, topography, ECEF,                  \
                          GeodeticRepresentation, HorizontalRepresentation, LTP
        import astropy.units as u
        globs = { "geomagnet": geomagnet, "store": store,
                  "topography": topography, "ECEF": ECEF, "LTP": LTP,
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

    sys.exit(exit_status)
