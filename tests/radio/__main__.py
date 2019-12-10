# -*- coding: utf-8 -*-
"""
Run all unit tests for the radio_simus package
"""
import os
import unittest
import sys


def suite():
    test_loader = unittest.TestLoader()
    path = os.path.dirname(__file__)
    test_suite = test_loader.discover(path, pattern="test_*.py")
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    r = not runner.run(suite()).wasSuccessful()
    sys.exit(r)
