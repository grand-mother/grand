import numpy as np
import pytest
from grand import CartesianRepresentation
from grand.aoi.antenna import Antenna #function a tester

def test_antenna_defaults():
    """Test default initialization types and shapes."""
    a = Antenna()

    # Check id and model types
    assert isinstance(a.id, int)
    assert isinstance(a.model, (int, float, object))

    # Check type of Cartesian fields
    assert isinstance(a._position, CartesianRepresentation)
    assert isinstance(a._tilt, CartesianRepresentation)
    assert isinstance(a._acceleration, CartesianRepresentation)

    # Check array types and shapes inside CartesianRepresentation
    for attr in ['x', 'y', 'z']:
        arr = getattr(a._position, attr)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1,)
        assert arr.dtype == np.float64


def test_position_setter_getter():
    """Test that setting position works correctly."""
    a = Antenna()
    new_pos = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
    a.position = new_pos

    # Check type
    assert isinstance(a.position, CartesianRepresentation)
    # Check each component’s shape and type
    for val in [a.position.x, a.position.y, a.position.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64

    # Check correctness of assignment
    assert np.allclose(a.position.x, new_pos[0])
    assert np.allclose(a.position.y, new_pos[1])
    assert np.allclose(a.position.z, new_pos[2])


def test_tilt_and_acceleration_setters():
    """Test tilt and acceleration setters for type and shape."""
    a = Antenna()
    new_vals = (np.array([0.1]), np.array([0.2]), np.array([0.3]))

    # Test tilt
    a.tilt = new_vals
    assert isinstance(a.tilt, CartesianRepresentation)
    for val in [a.tilt.x, a.tilt.y, a.tilt.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64

    # Test acceleration
    a.acceleration = new_vals
    assert isinstance(a.acceleration, CartesianRepresentation)
    for val in [a.acceleration.x, a.acceleration.y, a.acceleration.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64


@pytest.mark.parametrize("setter_name", ["position", "tilt", "acceleration"])
def test_setter_rejects_incorrect_shape(setter_name):
    """Ensure improper input raises error or fails gracefully."""
    a = Antenna()
    bad_input = (np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0]))
    with pytest.raises(Exception):
        setattr(a, setter_name, bad_input)
