"""
Unit tests for the grand.io.protocol module
"""

import unittest

from grand.io.protocol import InvalidBLOB, get
from tests import TestCase


class ProtocolTest(TestCase):
    """Unit tests for the protocol module"""

    def test_get(self):
        # Test the end-to-end chain
        blob = get("check.txt")
        self.assertEqual(blob, b"This is just a check\n")

        # Test the wrong url case
        with self.assertRaises(InvalidBLOB) as context:
            blob = get("toto")


if __name__ == "__main__":
    unittest.main()
