# -*- coding: utf-8 -*-
"""
Unit tests for the grand.libs.turtle module
"""

import os
import unittest

import numpy

import grand.store as store
from grand.libs import turtle


class TurtleTest(unittest.TestCase):
    """Unit tests for the turtle module"""

    def test_init(self):
        self.assertEqual(turtle.LIBNAME, "libturtle.so")
        self.assertNotEqual(turtle.LIBHASH, None)

    def test_install(self):
        self.assertTrue(os.path.exists(turtle.LIBPATH))


    def test_load(self):
        self.assertNotEqual(turtle._lib, None)

    def test_ecef(self):
        # Reference values
        ref = {
            "geodetic" : (45, 3, 1E+03),
            "ecef" : (4512105.81527233, 236469.44566852, 4488055.51564711),
            "horizontal" : (0, 90)}
        u = numpy.array(ref["ecef"])
        ref["direction"] = u / numpy.linalg.norm(u) # Assuming a spherical Earth

        # Check the ECEF to geodetic conversion
        ecef = turtle.ecef_from_geodetic(*ref["geodetic"])
        for i in range(3):
            self.assertAlmostEqual(ecef[i], ref["ecef"][i], 4)

        n = 10
        ecef = turtle.ecef_from_geodetic(n * (ref["geodetic"][0],),
            n * (ref["geodetic"][1],), n * (ref["geodetic"][2],))
        self.assertEqual(ecef.shape[0], n)
        self.assertEqual(ecef.shape[1], 3)
        for i in range(n):
            for j in range(3):
                self.assertAlmostEqual(ecef[i,j], ref["ecef"][j], 4)

        # Check the geodetic to ECEF conversion
        geodetic = turtle.ecef_to_geodetic(ref["ecef"])
        for i in range(3):
            self.assertAlmostEqual(geodetic[i], ref["geodetic"][i], 4)

        geodetic = turtle.ecef_to_geodetic(n * (ref["ecef"],))
        for i in range(3):
            self.assertEqual(geodetic[i].size, n) 
        for i in range(3):
            for j in range(10):
                self.assertAlmostEqual(geodetic[i][j], ref["geodetic"][i], 4)

        # Check the horizontal to ECEF conversion
        direction = turtle.ecef_from_horizontal(ref["geodetic"][0],
            ref["geodetic"][1], ref["horizontal"][0], ref["horizontal"][1])
        for i in range(3):
            self.assertAlmostEqual(direction[i], ref["direction"][i], 2)

        direction = turtle.ecef_from_horizontal(n * (ref["geodetic"][0],),
            n * (ref["geodetic"][1],), n * (ref["horizontal"][0],),
            n * (ref["horizontal"][1],))
        self.assertEqual(direction.shape[0], n)
        self.assertEqual(direction.shape[1], 3)
        for i in range(n):
            for j in range(3):
                self.assertAlmostEqual(direction[i,j], ref["direction"][j], 2)

        # Check the ECEF direction to horizontal conversion
        horizontal = turtle.ecef_to_horizontal(ref["geodetic"][0],
            ref["geodetic"][1], ref["direction"])
        self.assertAlmostEqual(horizontal[1], ref["horizontal"][1], 0)

        horizontal = turtle.ecef_to_horizontal(n * (ref["geodetic"][0],),
            n * (ref["geodetic"][1],), n * (ref["direction"],))
        for i in range(n):
            self.assertAlmostEqual(horizontal[1][i], ref["horizontal"][1], 0)


    def test_stack(self):
        # Fetch a test tile
        dirname, basename = "tests/topography", "N38E083.SRTMGL1.hgt"
        path = os.path.join(dirname, basename)
        if not os.path.exists(path):
            try:
                os.makedirs(dirname)
            except OSError:
                pass
            with open(path, "wb") as f:
                f.write(store.get(basename))

        # Check the stack initalisation
        stack = turtle.Stack(dirname)
        self.assertNotEqual(stack._stack, None)
        self.assertEqual(stack.path, dirname)
        self.assertEqual(stack.stack_size, 0)

        # Check the elevation getter for a single entry
        elevation = stack.elevation(38.5, 83.5)
        self.assertFalse(numpy.isnan(elevation))

        # Check the elevation getter for out of map entries
        elevation = stack.elevation(45.5, 3.5)
        self.assertTrue(numpy.isnan(elevation))

        # Check the elevation getter for vectorized entries
        n = 10
        elevation = stack.elevation(n * (38.5,), n * (83.5,))
        for i in range(n):
            self.assertFalse(numpy.isnan(elevation[i]))

        # Check the manual deletion
        del stack

        # Check the empty stack initalisation
        stack = turtle.Stack("")
        self.assertNotEqual(stack._stack, None)
        self.assertEqual(stack.path, "")
        self.assertEqual(stack.stack_size, 0)

        # Check the elevation getter for empty entries
        elevation = stack.elevation(45.5, 3.5)
        self.assertTrue(numpy.isnan(elevation))


    def test_map(self):
        # Check the map loading
        path = os.path.join(os.path.dirname(__file__), "data", "map.png")
        map_ = turtle.Map(path)
        self.assertNotEqual(map_._map, None)
        self.assertEqual(map_.path, path)

        # Check the elevation getter for a single entry
        elevation = map_.elevation(0, 0)
        self.assertTrue(1000)

        # Check the elevation getter for vectorized entries
        n = 10
        elevation = map_.elevation(n * (0,), n * (0,))
        for i in range(n):
            self.assertTrue(1000)

        # Check the manual deletion
        del map_

        # Check for invalid path case
        with self.assertRaises(RuntimeError) as context:
            map_ = turtle.Map("")
        self.assertRegex(context.exception.args[0], "^A TURTLE library error")


if __name__ == "__main__":
    unittest.main()
