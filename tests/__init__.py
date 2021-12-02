'''
Unit tests for the GRAND package
'''

import argparse
import doctest
import os
import unittest
import sys
from numbers import Number
import numpy

from pathlib import Path

__all__ = ['main']


class TestCase(unittest.TestCase):
    def assertArray(self, a, b, tol=9):
        '''Check that two numpy.ndarray are consistent'''
        if len(a.shape) > 1:
            a, b = a.flatten(), b.flatten()
        for i, ai in enumerate(a): self.assertAlmostEqual(ai, b[i], tol)

    #RK
    def assertQuantity(self, a, b, tol=9):
        '''Check that two quantities are consistent'''
        if isinstance(a, Number):
            self.assertAlmostEqual(a, b, tol)
        elif isinstance(a, (list, numpy.ndarray)):
            if isinstance(a, numpy.ndarray):
                n = a.size
                self.assertEqual(n, b.size)
            a, b = a.flatten(), b.flatten()
            for i, ai in enumerate(a): self.assertAlmostEqual(ai, b[i], tol)

    def assertCartesian(self, a, b, tol=9):
        '''Check that two CartesianRepresentations are consistent'''
        self.assertQuantity(a[0], b[0], tol)
        self.assertQuantity(a[1], b[1], tol)
        self.assertQuantity(a[2], b[2], tol)

    def assertSpherical(self, a, b, tol=9):
        '''Check that two SphericalRepresentations are consistent'''
        self.assertQuantity(a[0], b[0], tol)
        self.assertQuantity(a[1], b[1], tol)
        self.assertQuantity(a[2], b[2], tol)

def main():
    '''Run a local test suite
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbosity', help='set output verbosity', type=int, default=2)
    parser.add_argument(
        '--doc', help='run only doc tests', action='store_true', default=False)
    parser.add_argument(
        '--unit', help='run only unit tests', action='store_true',
        default=False)
    options = parser.parse_args()

    dirname = Path(__file__).parent
    localdir = Path(sys.argv[0]).parent
    exit_status = 0

    if not options.doc:
        # Look for unit tests
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(localdir, pattern='test_*.py')

        # Run the unit tests
        if test_suite.countTestCases():
            print('running unit tests...')
            if options.verbosity > 1:
                print(70 * '=')

            runner = unittest.TextTestRunner(verbosity=options.verbosity)
            exit_status = not runner.run(test_suite).wasSuccessful()
            if exit_status:
                sys.exit(exit_status)

    if not options.unit:
        # Look for doc tests
        #from astropy.coordinates import CartesianRepresentation
        #import astropy.units as u
        from grand import geomagnet, store, topography, ECEF, CartesianRepresentation, \
                          GeodeticRepresentation, HorizontalRepresentation, LTP
        import grand.io as io
        globs = { 'geomagnet': geomagnet, 'io': io, 'store': store,
                  'topography': topography,
                  'CartesianRepresentation': CartesianRepresentation,
                  'ECEF': ECEF, 'LTP': LTP,
                  'GeodeticRepresentation': GeodeticRepresentation,
                  'HorizontalRepresentation': HorizontalRepresentation}#, 'u': u}

        test_suite = unittest.TestSuite()
        top = dirname.parent
        relative = localdir.relative_to(dirname)
        for root, _, filenames in os.walk(top / 'docs' / relative):
            for filename in filenames:
                if filename[-4:] != '.rst':
                    continue
                path = os.path.join(root, filename)
                test_suite.addTest(doctest.DocFileSuite(path,
                        module_relative=False, globs=globs))

        # Run the doc tests
        if test_suite.countTestCases():
            print()
            print('running doc tests...')
            if options.verbosity > 1:
                print(70 * '=')

            runner = unittest.TextTestRunner(verbosity=options.verbosity)
            exit_status = not runner.run(test_suite).wasSuccessful()
            try:
                os.remove('data.hdf5')
            except FileNotFoundError:
                pass

    sys.exit(exit_status)
