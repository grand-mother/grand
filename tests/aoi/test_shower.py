import numpy as np
import pytest
from grand import CartesianRepresentation
from grand.aoi.shower import Shower  # adjust the import path if needed


def test_shower_defaults():
    """Test default initialization types and shapes."""
    s = Shower()

    # Check scalar attributes
    for attr in ['energy_em', 'energy_primary', 'Xmax', 'azimuth', 'zenith']:
        val = getattr(s, attr)
        assert isinstance(val, (int, float))
        assert val == 0

    # Check CartesianRepresentation fields
    assert isinstance(s._Xmaxpos, CartesianRepresentation)
    assert isinstance(s._origin_geoid, CartesianRepresentation)
    assert isinstance(s._core_ground_pos, CartesianRepresentation)

    # Check shapes and dtypes
    for cr in [s._Xmaxpos, s._origin_geoid, s._core_ground_pos]:
        for attr in ['x', 'y', 'z']:
            arr = getattr(cr, attr)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (1,)
            assert arr.dtype == np.float64


@pytest.mark.parametrize(
    "setter_name,values",
    [
        ("Xmaxpos", (np.array([1.0]), np.array([2.0]), np.array([3.0]))),
        ("origin_geoid", (np.array([0.1]), np.array([0.2]), np.array([0.3]))),
        ("core_ground_pos", (np.array([5.0]), np.array([6.0]), np.array([7.0]))),
    ],
)
def test_cartesian_setters(setter_name, values):
    """Test that the property setters correctly assign values."""
    s = Shower()
    setattr(s, setter_name, values)
    cr = getattr(s, setter_name)

    assert isinstance(cr, CartesianRepresentation)

    for val in [cr.x, cr.y, cr.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64

    np.testing.assert_allclose(cr.x, values[0])
    np.testing.assert_allclose(cr.y, values[1])
    np.testing.assert_allclose(cr.z, values[2])


@pytest.mark.parametrize("setter_name", ["Xmaxpos", "origin_geoid", "core_ground_pos"])
def test_setters_reject_incorrect_shape(setter_name):
    """Ensure improper input raises an error or fails gracefully."""
    s = Shower()
    bad_input = (np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0]))
    with pytest.raises(Exception):
        setattr(s, setter_name, bad_input)
