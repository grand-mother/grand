from grand.aoi.antenna import Antenna #function a tester
import os
import numpy as np
import unittest
from grand import CartesianRepresentation

if os.path.isfile("dummy_data.root"):
    os.remove("dummy_data.root")

os.system("python examples/dataio/data_storing.py dummy_data.root")


class TestAntenna(unittest.TestCase):

    def setUp(self):
        """Create a default Antenna object for each test."""
        self.a = Antenna()

    def test_default_initialization(self):
        """Test default initialization types and shapes."""
        a = self.a

        # Check id and model types
        self.assertIsInstance(a.id, int)
        self.assertIsInstance(a.model, (int, float, object))

        # Check Cartesian fields
        self.assertIsInstance(a._position, CartesianRepresentation)
        self.assertIsInstance(a._tilt, CartesianRepresentation)
        self.assertIsInstance(a._acceleration, CartesianRepresentation)

        # Check arrays inside CartesianRepresentation
        for attr in ['x', 'y', 'z']:
            arr = getattr(a._position, attr)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (1,))
            self.assertEqual(arr.dtype, np.float64)

    def test_position_setter_and_getter(self):
        """Test setting and getting the position property."""
        new_pos = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        self.a.position = new_pos

        pos = self.a.position
        self.assertIsInstance(pos, CartesianRepresentation)

        for val in [pos.x, pos.y, pos.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

        np.testing.assert_allclose(pos.x, new_pos[0])
        np.testing.assert_allclose(pos.y, new_pos[1])
        np.testing.assert_allclose(pos.z, new_pos[2])

    def test_tilt_and_acceleration_setters(self):
        """Test tilt and acceleration property setters."""
        new_vals = (np.array([0.1]), np.array([0.2]), np.array([0.3]))

        # Test tilt
        self.a.tilt = new_vals
        tilt = self.a.tilt
        self.assertIsInstance(tilt, CartesianRepresentation)
        for val in [tilt.x, tilt.y, tilt.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

        # Test acceleration
        self.a.acceleration = new_vals
        acc = self.a.acceleration
        self.assertIsInstance(acc, CartesianRepresentation)
        for val in [acc.x, acc.y, acc.z]:
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val.dtype, np.float64)

    def test_setter_rejects_incorrect_shape(self):
        """Ensure improper input raises an error or fails gracefully."""
        bad_input = (np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0]))
        for prop in ["position", "tilt", "acceleration"]:
            with self.assertRaises(Exception):
                setattr(self.a, prop, bad_input)


if __name__ == "__main__":
    unittest.main()
