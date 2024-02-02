"""
Unit test for basis/traces_event.py
"""

from grand.basis.traces_event import *


def parabole(v_x):
    return -10 * v_x * v_x + 20 * v_x + 30


def get_tr3d(nb_sample=4):
    tr3d = Handling3dTraces()
    size_tot = 3 * 3 * nb_sample
    traces = np.arange(size_tot, dtype=np.float32).reshape((3, 3, nb_sample))
    du_id = np.arange(3)
    t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    return tr3d


def get_tr3d_du(nb_du=2, nb_sample=4):
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
    tr3d = get_tr3d_du(nb_du, 10)
    tr3d.traces[:, :, :] = 0.0
    tr3d.traces[:, 0, :] = 0
    tr3d.traces[:, 1, :] = 3.0
    tr3d.traces[:, 2, :] = 4.0
    tr3d.traces[0, :, :] = 0
    tr3d.traces[0, 1, :] = 10
    max_norm = tr3d.get_max_norm()
    res = np.ones(nb_du) * 5
    res[0] = 10
    assert np.allclose(max_norm, res)


def test_get_norm():
    nb_du = 5
    nb_sple = 8
    tr3d = get_tr3d_du(nb_du, nb_sple)
    tr3d.traces[:, :, :] = 0.0
    tr3d.traces[:, 0, :] = 0
    tr3d.traces[:, 1, :] = 3.0
    tr3d.traces[:, 2, :] = 4.0
    tr3d.traces[0, :, :] = 0
    tr3d.traces[0, 1, :] = 10
    norm = tr3d.get_norm()
    assert norm.shape == (nb_du, nb_sple)
    res = np.ones(nb_sple) * 10
    assert np.allclose(norm[0], res)
    res1 = np.ones(nb_sple) * 5
    assert np.allclose(norm[1], res1)


def test_get_tmax_vmax_1():
    tr3d = get_tr3d(32)
    tr3d.traces[:, :, :] = 0
    tr3d.traces[0, 1, 0] = 3.0
    tr3d.traces[1, 2, 2] = 5.0
    tr3d.traces[2, 0, 3] = 7.0
    tmax, vmax = tr3d.get_tmax_vmax(False, "no")
    t_tmax = np.array(
        [
            tr3d.t_start_ns[0],
            tr3d.t_start_ns[1] + tr3d.get_delta_t_ns()[0] * 2,
            tr3d.t_start_ns[2] + tr3d.get_delta_t_ns()[0] * 3,
        ]
    )
    assert np.allclose(tmax, t_tmax)
    t_vmax = np.array([3.0, 5.0, 7.0])
    assert np.allclose(vmax, t_vmax)
    #
    n_sig = tr3d.get_size_trace()
    v_x = np.linspace(0, 3, n_sig)
    epsilon = 0.001
    v_y = parabole(v_x) + np.random.normal(0, epsilon, n_sig)
    idx = np.argmax(v_y)
    print(idx, v_y[idx])

    tmax, vmax = tr3d.get_tmax_vmax(False)
    t_tmax = np.array(
        [
            tr3d.t_start_ns[0],
            tr3d.t_start_ns[1] + tr3d.get_delta_t_ns()[0] * 2,
            tr3d.t_start_ns[2] + tr3d.get_delta_t_ns()[0] * 3,
        ]
    )
    assert np.allclose(tmax, t_tmax)
    t_vmax = np.array([3.0, 5.0, 7.0])
    assert np.allclose(vmax, t_vmax)


def test_get_max_abs():
    tr3d = get_tr3d()
    tr3d.traces[:, :, :] = 0
    tr3d.traces[0, 1, :] = -1.0
    tr3d.traces[0, 1, 0] = -3.0
    tr3d.traces[1, 2, 2] = -5.0
    tr3d.traces[2, 0, 3] = -7.0
    vmax = tr3d.get_max_abs()
    t_vmax = np.array([3.0, 5.0, 7.0])
    assert np.allclose(vmax, t_vmax)


def test_get_extended_traces():
    tr3d = get_tr3d(4)
    tr3d.traces[2, 0, 0] = 123
    common_time, extended_traces = tr3d.get_extended_traces()
    t_ct = np.arange(100, 124)
    assert np.allclose(common_time, t_ct)
    assert extended_traces[2, 0, 20] == tr3d.traces[2, 0, 0]


def test_keep_only_trace_with_ident():
    nb_du = 5
    nb_sample = 10
    tr3d = Handling3dTraces()
    size_tot = nb_du * 3 * nb_sample
    traces = np.arange(size_tot, dtype=np.float32).reshape((nb_du, 3, nb_sample))
    print(traces)
    du_id = ["A1", "T1", "A2", "T2", "A3"]
    t_start_ns = np.arange(nb_du, dtype=np.float32) * 10 + 100
    f_samp_mhz = 1000
    tr3d.init_traces(traces, du_id, t_start_ns, f_samp_mhz)
    print(tr3d.idx2idt)
    print(tr3d.idt2idx)
    tr3d.keep_only_trace_with_ident(["T1", "T2"])
    assert tr3d.get_nb_trace() == 2
    assert tr3d.traces[0, 0, 0] == 30
    assert tr3d.traces[1, 0, 0] == 90


def test_get_copy():
    # new_traces
    new_traces = np.ones(G_tr3d.traces.shape, dtype=np.float32)
    new_traces[1, 1, 1] = 321
    my_tr3d = G_tr3d.get_copy(new_traces)
    val = 123
    my_tr3d.traces[0, 0, 0] = val
    assert G_tr3d.traces[0, 0, 0] != val
    assert my_tr3d.traces[1, 1, 1] == 321
    assert my_tr3d.traces[1, 1, 0] == 1
    # no new_traces
    my_tr3d = G_tr3d.get_copy()
    assert np.allclose(G_tr3d.traces, my_tr3d.traces)
    val = 123
    ori_val = G_tr3d.traces[1, 1, 1]
    my_tr3d.traces[1, 1, 1] = 321
    assert G_tr3d.traces[1, 1, 1] == ori_val
    # no new_traces, rza traces
    my_tr3d = G_tr3d.get_copy(0)
    assert np.allclose(my_tr3d.traces, np.zeros(my_tr3d.traces.shape))


def test_remove_trace_low_signal():
    tr3d = get_tr3d_du(5, 20)
    tr3d.traces[0] = 1
    tr3d.traces[1] = 10
    tr3d.traces[2] = 0.5
    tr3d.traces[3] = 11
    tr3d.traces[4] = 1
    ori_tr3d = tr3d.get_copy()
    tr3d.remove_trace_low_signal(5)
    assert tr3d.get_nb_trace() == 2
    tr3d = ori_tr3d.get_copy()
    norm = tr3d.get_max_norm()
    tr3d.remove_trace_low_signal(5, norm)
    assert tr3d.get_nb_trace() == 2
    tr3d = ori_tr3d.get_copy()
    noise = np.random.normal(0, 1, tr3d.traces.size).reshape(tr3d.traces.shape)
    tr3d.traces += noise
    # print( tr3d.traces)
    tm, norm = tr3d.get_tmax_vmax(False, "no")
    # print(norm)
    l_idx = tr3d.remove_trace_low_signal(5, norm)
    assert tr3d.get_nb_trace() == 2
    assert np.allclose(l_idx, [1, 3])


def notest_downsize_sampling():
    n_size = 128
    tr3d = get_tr3d_du(3, 128)
    tr3d.trace = np.random.normal(0, 1, tr3d.traces.size).reshape(tr3d.traces.shape)
    ori = tr3d.get_copy()
    tr3d.downsize_sampling(2)
    assert tr3d.get_size_trace() == n_size / 2
    print()
    print(tr3d.traces[0, 0, :10])
    print()
    print(ori.trace[0, 0, :20])
    assert np.allclose(tr3d.traces[0], ori.trace[0, :, ::2])
    assert np.allclose(tr3d.traces[1], ori.trace[1, :, ::2])
    assert np.allclose(tr3d.traces[2], ori.trace[2, :, ::2])
