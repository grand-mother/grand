# -*- coding: utf-8 -*-
"""
Unit tests for the grand.libs.tools module
"""

import os
import unittest

import grand.libs.tools as tools
from grand.libs import LIBDIR


class ToolsTest(unittest.TestCase):
    """Unit tests for the tools module"""

    def test_meta(self):
        meta = tools.Meta("test-tools")
        meta["LIBHASH"] = "abcd"
        meta.update()
        del meta

        meta = tools.Meta("test-tools")
        self.assertEqual(meta["LIBHASH"], "abcd")
        os.remove(os.path.join(LIBDIR, ".test-tools.json"))


    def test_temporary(self):
        path = os.getcwd()
        with tools.Temporary(
            "https://github.com/grand-mother/libs") as _:
            self.assertNotEqual(path, os.getcwd())
        self.assertEqual(path, os.getcwd())


if __name__ == "__main__":
    unittest.main()
