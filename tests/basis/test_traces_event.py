"""

"""

import numpy as np

from grand.basis.traces_event import Handling3dTracesOfEvent

def get_tr3d():
    tr3d = Handling3dTracesOfEvent()
    size_tot = 3 * 3 * 4
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, 4))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    return tr3d


def test_init_traces():
    tr3d = Handling3dTracesOfEvent()
    size_tot = 3 * 3 * 4
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, 4))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    assert tr3d.traces[0, 0, 0] == 0.0
    assert tr3d.traces[2, 2, 3] == size_tot - 1
    assert tr3d.du_id[-1] == 2
    assert np.allclose(tr3d.t_start_ns, np.array([100, 110, 120]))


def test_define_t_samples():
    tr3d = Handling3dTracesOfEvent()
    size_tot = 3 * 3 * 4
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, 4))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 2000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    assert np.allclose(tr3d.t_samples[0], np.array([100.0, 100.5, 101.0, 101.5]))
    tr3d.f_samp_mhz = 1000
    tr3d.t_samples = np.zeros(0)
    tr3d._define_t_samples()
    assert np.allclose(tr3d.t_samples[-1], np.array([120.0, 121, 122.0, 123]))
    

def test_delta_t_ns():
    tr3d = get_tr3d()
    assert tr3d.delta_t_ns() == 1.0

