"""
Unit test for basis/traces_event.py
"""

from grand.basis.traces_event import *


def get_tr3d(nb_sample=4):
    tr3d = Handling3dTraces()
    size_tot = 3 * 3 * nb_sample
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, nb_sample))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    return tr3d

def get_tr3d_du(nb_du = 2, nb_sample=4):
    tr3d = Handling3dTraces()
    size_tot = nb_du * 3 * nb_sample
    traces = np.arange(size_tot, dtype=np.float32).reshape((nb_du, 3, nb_sample))
    du_id = np.arange(nb_du)
    t_start_ns = np.arange(nb_du, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    return tr3d


G_tr3d = get_tr3d(10)


def test_init_traces():
    tr3d = Handling3dTraces()
    size_tot = 3 * 3 * 4
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, 4))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    assert tr3d.traces[0, 0, 0] == 0.0
    assert tr3d.traces[2, 2, 3] == size_tot - 1
    assert tr3d.idx2idt[-1] == 2
    assert np.allclose(tr3d.t_start_ns, np.array([100, 110, 120]))


def test_define_t_samples():
    tr3d = Handling3dTraces()
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


def test_get_delta_t_ns():
    assert np.allclose(G_tr3d.get_delta_t_ns(), 1.0)


def test_set_unit_axis():
    G_tr3d.set_unit_axis(axis_name="dir")
    assert G_tr3d.axis_name[0] == "SN"
    s_unit = "toto"
    G_tr3d.set_unit_axis(s_unit)
    assert G_tr3d.unit_trace == s_unit
    s_type = "tutu"
    G_tr3d.set_unit_axis(type_tr=s_type)
    assert G_tr3d.type_trace == s_type


def test_reduce_nb_du():
    tr3d = get_tr3d()
    pos_du = np.arange(3 * tr3d.get_nb_trace()).reshape((-1, 3))
    # print(pos_du)
    tr3d.init_network(pos_du)
    tr3d.reduce_nb_trace(1)
    assert tr3d.get_nb_trace() == 1
    assert tr3d.network.get_nb_du() == 1


def test_get_max_norm():
    nb_du = 5
    tr3d = get_tr3d_du(nb_du,10)
    tr3d.traces[:, :, :] = 0.0
    tr3d.traces[:, 0, :] = 0
    tr3d.traces[:, 1, :] = 3.0
    tr3d.traces[:, 2, :] = 4.0
    tr3d.traces[0, :, :] = 0
    tr3d.traces[0, 1, :] = 10
    max_norm = tr3d.get_max_norm()
    res =np.ones(nb_du)*5
    res[0] = 10
    assert np.allclose(max_norm, res)


#
# class EventTracesTest(TestCase):
#     def __init__(self, *args):
#         super().__init__(*args)
#         self.tr3d = Handling3dTraces()
#         self.size_tot = 3 * 3 * 4
#         self.traces = np.arange(self.size_tot, dtype=np.float32).reshape((3, 3, 4))
#         self.du_id = np.arange(3)
#         self.t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
#         self.f_samp_mhz = 2000
#         self.tr3d.init_traces(self.traces, self.du_id, self.t_start_ns, self.f_samp_mhz)
#
#     def get_traces(self):
#         self.size_tot = 3 * 3 * 4
#         self.traces = np.arange(self.size_tot, dtype=np.float32).reshape((3, 3, 4))
#         self.du_id = np.arange(3)
#         self.t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
#         self.f_samp_mhz = 2000
#         self.tr3d.init_traces(self.traces, self.du_id, self.t_start_ns, self.f_samp_mhz)
#
#

#
#
#
#


#
#
#     def test_get_norm(self):
#         # RK: add more/better tests.
#         trnorm = self.tr3d.get_norm() # (nb_du, n_traces)
#         assert trnorm.shape==(len(self.du_id), self.traces.shape[-1])
#
#     def test_get_tmax_vmax(self):
#         # RK: add more/better tests.
#         tmax, vmax = self.tr3d.get_tmax_vmax()
#         assert len(tmax)==len(self.du_id)
#         assert len(vmax)==len(self.du_id)
#
#     def test_get_nb_du(self):
#         # RK: add more/better tests.
#         nb_du = self.tr3d.get_nb_trace()
#         assert nb_du==len(self.du_id)
#
#     def test_get_size_trace(self):
#         # RK: add more/better tests.
#         assert self.tr3d.get_size_trace()==self.traces.shape[-1]
#
#     def test_get_extended_traces(self):
#         # RK: add more/better tests.
#         common_time, extended_traces = self.tr3d.get_extended_traces()
#         assert extended_traces.shape[0]==len(self.du_id)
#
# if __name__ == "__main__":
#     unittest.main()
