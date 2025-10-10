import unittest
import numpy as np

from grand import CartesianRepresentation
from grand.aoi.shower import Shower  # adjust import path if needed


class TestShower(unittest.TestCase):

    def setUp(self):
        """Create a default Shower object for each test."""
        self.s = Shower()

    def test_default_initialization(self):
        """Test default initialization types and shapes."""
        s = self.s

        # Check scalar fields
        for attr in ['energy_em', 'energy_primary', 'Xmax', 'azimuth', 'zenith']:
            val = getattr(s, attr)
            self.assertIsInstance(val, (int, float))
            self.assertEqual(val, 0)

        # Check CartesianRepresentation fields
        self.assertIsInstance(s._Xmaxpos, CartesianRepresentation)
        self.assertIsInstance(s._origin_geoid, CartesianRepresentation)
        self.assertIsInstance(s._core_ground_pos, CartesianRepresentation)

        # Check arrays inside CartesianRepresentation
        for cr in [s._Xmaxpos, s._origin_geoid, s._core_ground_pos]:
            for attr in ['x', 'y', 'z']:
                arr = getattr(cr, attr)
                self.assertIsInstance(arr, np.ndarray)
                self.assertEqual(arr.shape, (1,))
                self.assertEqual(arr.dtype, np.float64)

    def test_Xmaxpos_setter_and_getter(self):
        """Test Xmaxpos setter and getter correctness."""
        new_val = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        self.s.Xmaxpos = new_val
        pos = self.s.Xmaxpos

        self.assertIsInstance(pos, CartesianRepresentation)
        for val in [pos.x, pos.y, pos.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

        np.testing.assert_allclose(pos.x, new_val[0])
        np.testing.assert_allclose(pos.y, new_val[1])
        np.testing.assert_allclose(pos.z, new_val[2])

    def test_origin_geoid_setter_and_getter(self):
        """Test origin_geoid setter and getter correctness."""
        new_val = (np.array([0.1]), np.array([0.2]), np.array([0.3]))
        self.s.origin_geoid = new_val
        geo = self.s.origin_geoid

        self.assertIsInstance(geo, CartesianRepresentation)
        for val in [geo.x, geo.y, geo.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

        np.testing.assert_allclose(geo.x, new_val[0])
        np.testing.assert_allclose(geo.y, new_val[1])
        np.testing.assert_allclose(geo.z, new_val[2])

    def test_core_ground_pos_setter_and_getter(self):
        """Test core_ground_pos setter and getter correctness."""
        new_val = (np.array([5.0]), np.array([6.0]), np.array([7.0]))
        self.s.core_ground_pos = new_val
        core = self.s.core_ground_pos

        self.assertIsInstance(core, CartesianRepresentation)
        for val in [core.x, core.y, core.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

        np.testing.assert_allclose(core.x, new_val[0])
        np.testing.assert_allclose(core.y, new_val[1])
        np.testing.assert_allclose(core.z, new_val[2])

    def test_setters_reject_incorrect_shape(self):
        """Ensure improper inputs raise an error or fail gracefully."""
        bad_input = (np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0]))
        for prop in ["Xmaxpos", "origin_geoid", "core_ground_pos"]:
            with self.assertRaises(Exception):
                setattr(self.s, prop, bad_input)


if __name__ == "__main__":
    unittest.main()
