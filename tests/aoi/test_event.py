import numpy as np
import pytest
from grand.aoi.event import Event
from grand.aoi.timetrace import Voltage

def test_event_defaults():
    """Test default initialization types and shapes for Event."""
    e = Event()

    # Basic scalar fields
    assert e.event_number is None
    assert e.run_number is None
    assert isinstance(e.is_reconstructed, bool)
    assert isinstance(e.is_wave, bool)
    assert isinstance(e.is_eas, bool)

    # Time vector default
    assert isinstance(e.t_vector, np.ndarray)
    assert e.t_vector.shape == (1,)
    assert e.t_vector.dtype == np.float32

    # Origin geoid defaults
    assert e.origin_geoid is not None
    for attr in ['x', 'y', 'z']:
        arr = getattr(e.origin_geoid, attr)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1,)
        assert arr.dtype == np.float64

    # Internal placeholders
    assert e.antennas is None
    assert e.voltages is None
    assert e.efields is None

    # Other defaults
    assert isinstance(e.run_mode, (np.uint32, int))
    # file is stringified
    assert isinstance(e.file, str)


def test_event_origin_geoid_setter_getter():
    """Test that setting origin_geoid works correctly."""
    e = Event()
    new_origin = (np.array([10.0], dtype=np.float64),
                  np.array([20.0], dtype=np.float64),
                  np.array([30.0], dtype=np.float64))
    e.origin_geoid = new_origin

    assert isinstance(e.origin_geoid, type(e._origin_geoid))
    for val in [e.origin_geoid.x, e.origin_geoid.y, e.origin_geoid.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64

    assert np.allclose(e.origin_geoid.x, new_origin[0])
    assert np.allclose(e.origin_geoid.y, new_origin[1])
    assert np.allclose(e.origin_geoid.z, new_origin[2])


def test_fill_t_vector_rectangular():
    """fill_t_vector with same-length t_vectors (rectangular array)."""
    e = Event()

    v1 = Voltage()
    v1.t_vector = np.array([0, 2, 4], dtype=np.int64)

    v2 = Voltage()
    v2.t_vector = np.array([0, 2, 6], dtype=np.int64)

    e.voltages = [v1, v2]

    e.fill_t_vector(resolution=1)
    assert isinstance(e.t_vector, np.ndarray)
    assert np.array_equal(e.t_vector, np.arange(0, 6 + 1, 1))


def test_fill_t_vector_irregular():
    """fill_t_vector with different-length t_vectors (object array path)."""
    e = Event()

    v1 = Voltage()
    v1.t_vector = np.array([0, 3], dtype=np.int64)

    v2 = Voltage()
    v2.t_vector = np.array([1, 2, 5], dtype=np.int64)

    e.voltages = [v1, v2]

    e.fill_t_vector(resolution=1)
    assert np.array_equal(e.t_vector, np.arange(0, 5 + 1, 1))


def test_get_voltage_at_time_stacks_values():
    """get_voltage_at_time returns stacked values from all voltages."""
    e = Event()

    v1 = Voltage()
    v2 = Voltage()

    # Monkeypatch instance methods to avoid building full trace structures
    v1.get_value_at_time = lambda t: np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2.get_value_at_time = lambda t: np.array([4.0, 5.0, 6.0], dtype=np.float32)

    e.voltages = [v1, v2]
    out = e.get_voltage_at_time(123456789)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1.0, 2.0, 3.0])
    assert np.allclose(out[1], [4.0, 5.0, 6.0])


def test_get_hilbert_voltage_at_time_stacks_values():
    """get_hilbert_voltage_at_time stacks hilbert values from all voltages."""
    e = Event()

    v1 = Voltage()
    v2 = Voltage()

    v1.get_hilbert_value_at_time = lambda t: np.array([10.0, 20.0, 30.0], dtype=np.float32)
    v2.get_hilbert_value_at_time = lambda t: np.array([40.0, 50.0, 60.0], dtype=np.float32)

    e.voltages = [v1, v2]
    out = e.get_hilbert_voltage_at_time(42)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [10.0, 20.0, 30.0])
    assert np.allclose(out[1], [40.0, 50.0, 60.0])


class DummyTRun:
    def __init__(self):
        self.last_call = None
        self.run_mode = np.uint32(7)
        self.data_source = "detector"
        self.data_generator = "GEN"
        self.data_generator_version = "v1.2.3"
        self.site = "DummySite"
        self.origin_geoid = (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        )
        self.t_bin_size = np.array([4], dtype=np.int32)
        self.site_layout = "star_shape"

    def get_run(self, run_number):
        self.last_call = ("get_run", run_number)
        return 1

    def get_entry(self, entry_number):
        self.last_call = ("get_entry", entry_number)
        return 1


class DummyTRunRawVoltage:
    def __init__(self):
        self.last_call = None

    def get_run(self, run_number):
        self.last_call = ("get_run", run_number)
        return 1

    def get_entry(self, entry_number):
        self.last_call = ("get_entry", entry_number)
        return 1


def test_fill_event_from_runtree_copies_fields_and_starshape():
    """fill_event_from_runtree reads from provided trun and copies fields."""
    e = Event()
    e.trun = DummyTRun()

    # Case: default, run_entry_number is None, run_number is None -> uses entry 0
    ret = e.fill_event_from_runtree(run_entry_number=None)
    assert ret == 1
    assert e.trun.last_call == ("get_entry", 0)

    # Fields copied correctly
    assert e.run_mode == np.uint32(7)
    assert e.data_source == "detector"
    assert e.data_generator == "GEN"
    assert e.data_generator_version == "v1.2.3"
    assert e.site == "DummySite"
    assert e._t_bin_size == 4
    assert e.is_starshape is True

    # origin_geoid copied
    for attr, expected in zip(['x', 'y', 'z'], e.trun.origin_geoid):
        arr = getattr(e.origin_geoid, attr)
        assert np.allclose(arr, expected)


def test_fill_event_from_runrawvoltagetree_starshape_uses_entry_number():
    """fill_event_from_runrawvoltagetree uses _entry_number if starshape and run unset."""
    e = Event()
    e.is_starshape = True
    e._entry_number = 5
    e.trunrawvoltage = DummyTRunRawVoltage()

    ret = e.fill_event_from_runrawvoltagetree(run_entry_number=None)
    assert ret == 1
    assert e.trunrawvoltage.last_call == ("get_entry", 5)


def test_fill_run_tree_raises_treeexists_when_trun_present():
    """fill_run_tree should raise TreeExists if trun already initialised and not overwriting."""
    e = Event()
    e.trun = object()  # any non-None sentinel
    with pytest.raises(Exception):
        e.fill_run_tree(overwrite=False)


def test_fill_voltage_tree_raises_treeexists_when_tvoltage_present():
    """fill_voltage_tree should raise TreeExists if tvoltage already initialised and not overwriting."""
    e = Event()
    e.tvoltage = object()
    with pytest.raises(Exception):
        e.fill_voltage_tree(overwrite=False)


def test_fill_efield_tree_raises_treeexists_when_tefield_present():
    """fill_efield_tree should raise TreeExists if tefield already initialised and not overwriting."""
    e = Event()
    e.tefield = object()
    with pytest.raises(Exception):
        e.fill_efield_tree(overwrite=False)