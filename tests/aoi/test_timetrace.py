import numpy as np
import pytest
from scipy.signal import hilbert
from grand import CartesianRepresentation
from grand.aoi.timetrace import Timetrace3D, Voltage, Efield

def test_timetrace_defaults():
    t = Timetrace3D()

    # Basic types
    assert isinstance(t._trace, CartesianRepresentation)
    assert isinstance(t._hilbert_trace, CartesianRepresentation)
    assert isinstance(t.t_vector, np.ndarray)
    assert isinstance(t.t0, np.datetime64)
    assert isinstance(t.trigger_time, np.datetime64)
    assert isinstance(t.du_id, int)

    # Trace defaults: empty float32 arrays
    for arr in (t._trace.x, t._trace.y, t._trace.z):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0,)
        assert arr.dtype == np.float32

    # Hilbert trace defaults: empty float32 arrays
    for arr in (t._hilbert_trace.x, t._hilbert_trace.y, t._hilbert_trace.z):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0,)
        assert arr.dtype == np.float32

    # t_vector default
    assert isinstance(t.t_vector, np.ndarray)
    assert t.t_vector.shape == (1,)
    assert t.t_vector.dtype == np.float32


def test_trace_setter_getter():
    t = Timetrace3D()
    x = np.array([0, 1, 2, 3], dtype=np.float32)
    y = np.array([10, 11, 12, 13], dtype=np.float32)
    z = np.array([20, 21, 22, 23], dtype=np.float32)

    t.trace = (x, y, z)

    assert isinstance(t.trace, CartesianRepresentation)
    for comp in (t.trace.x, t.trace.y, t.trace.z):
        assert isinstance(comp, np.ndarray)
        assert comp.shape == (4,)
        assert comp.dtype == np.float32

    assert np.allclose(t.trace.x, x)
    assert np.allclose(t.trace.y, y)
    assert np.allclose(t.trace.z, z)


def test_calculate_t_vector_and_get_value():
    t = Timetrace3D()
    x = np.array([0, 1, 2, 3], dtype=np.float32)
    y = x + 10
    z = x + 20
    t.trace = (x, y, z)

    # t_bin_size default is 2 ns; choose offset so start at 8
    t.t0 = np.datetime64(100, 'ns')
    time_offset = np.datetime64(92, 'ns')
    t.calculate_t_vector(time_offset)

    expected_t = np.arange(x.size) * t.t_bin_size + 8
    assert np.allclose(t.t_vector, expected_t)

    # Value present
    present_time = 10  # corresponds to index 1
    val = t.get_value_at_time(present_time)
    assert isinstance(val, np.ndarray)
    assert val.shape == (3,)
    assert np.allclose(val, np.array([x[1], y[1], z[1]], dtype=np.float32))

    # Value absent
    absent_time = 999
    val_absent = t.get_value_at_time(absent_time)
    assert isinstance(val_absent, np.ndarray)
    assert val_absent.shape == (3,)
    assert val_absent.dtype == np.float32
    assert np.allclose(val_absent, np.zeros(3, np.float32))


def test_hilbert_trace_and_get_hilbert_value_at_time():
    t = Timetrace3D()
    n = 64
    ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.sin(ts).astype(np.float32)
    y = (2.0 * np.sin(ts)).astype(np.float32)
    z = (0.5 * np.sin(ts)).astype(np.float32)
    t.trace = (x, y, z)

    # Trigger hilbert computation
    ht = t.hilbert_trace
    assert isinstance(ht, CartesianRepresentation)

    hx = np.abs(hilbert(x))
    hy = np.abs(hilbert(y))
    hz = np.abs(hilbert(z))

    # Shapes and dtypes (hilbert envelope is float64)
    for comp, exp in ((ht.x, hx), (ht.y, hy), (ht.z, hz)):
        assert isinstance(comp, np.ndarray)
        assert comp.shape == (n,)
        assert comp.dtype == np.float64
        assert np.allclose(comp, exp, rtol=1e-6, atol=1e-8)

    # Time vector and retrieval
    t.t0 = np.datetime64(0, 'ns')
    t.calculate_t_vector(np.datetime64(0, 'ns'))  # [0,2,4,...]
    query_time = 4  # index 2
    hv = t.get_hilbert_value_at_time(query_time)
    assert hv.shape == (3,)
    assert np.allclose(hv, np.array([hx[2], hy[2], hz[2]]))

    # Absent time -> zeros float32
    hv_absent = t.get_hilbert_value_at_time(9999)
    assert hv_absent.shape == (3,)
    assert hv_absent.dtype == np.float32
    assert np.allclose(hv_absent, np.zeros(3, np.float32))


@pytest.mark.parametrize("bad_trace", [
    (np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32)),
    (np.array([1.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32), np.array([5.0], dtype=np.float32)),
    (np.array([1.0], dtype=np.float32), np.array([3.0], dtype=np.float32), np.array([5.0, 6.0], dtype=np.float32)),
])
def test_trace_setter_rejects_incorrect_shape(bad_trace):
    t = Timetrace3D()
    with pytest.raises(Exception):
        t.trace = bad_trace


def test_voltage_defaults():
    v = Voltage()
    # Inherits Timetrace3D defaults
    assert isinstance(v._trace, CartesianRepresentation)
    assert isinstance(v._hilbert_trace, CartesianRepresentation)
    # Voltage-specific
    assert isinstance(v.is_triggered, bool)
    assert v.is_triggered is True


def test_efield_defaults():
    e = Efield()
    # Inherits Timetrace3D defaults
    assert isinstance(e._trace, CartesianRepresentation)
    assert isinstance(e._hilbert_trace, CartesianRepresentation)
    # Efield-specific
    assert isinstance(e.eta, (int, float))
    assert isinstance(e.a_ratio, (int, float))
    assert e.eta == 0
    assert e.a_ratio == 0