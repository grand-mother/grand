from grand.basis.signal import *
import grand.basis.traces_event as tre
import scipy as scp

G_FlagPlot = True

def test_get_peakamptime_norm_hilbert():
    '''
    Test :
    sinus apodisé, 0 est au milieu, et vaut 0
    par contre sa transformé d'Hilbert est maximum au milieu
    '''
    tr3d = tre.Handling3dTraces()
    nb_s = 1024
    ar_tr = np.zeros((2, 3, nb_s), dtype=np.float32)
    tr3d.init_traces(ar_tr, f_samp_mhz=1)
    v_x, step = np.linspace(-20, 20, nb_s, retstep=True)
    # v_y = np.sinc(v_x)
    # ar_tr[0, 0] = v_y
    # ar_tr[0, 1] = v_y * 0.5
    # ar_tr[0, 2] = v_y * 2
    v_y = np.sin(v_x) * scp.signal.windows.blackman(nb_s)
    ar_tr[1, 0] = v_y
    ar_tr[1, 1] = v_y * 2
    ar_tr[1, 2] = v_y * 3
    t_max, v_max, idx_max, norm_hilbert_amp = get_peakamptime_norm_hilbert(
        tr3d.t_samples, tr3d.traces
    )
    zero_pos_in_trace = (20 / step) * tr3d.t_samples[1, 1]
    if G_FlagPlot:
        # tr3d.plot_trace_du(0)
        tr3d.plot_trace_du(1)
        tre.plt.vlines(zero_pos_in_trace, -5, 5)
        tre.plt.plot(tr3d.t_samples[1], norm_hilbert_amp[1])
        print(t_max, zero_pos_in_trace, idx_max, v_max)
    true_t_max = zero_pos_in_trace
    delta_t = tr3d.t_samples[1, 1]
    # sin*window is close to zero à t_max
    assert np.allclose(tr3d.traces[1, :, idx_max[1]], np.zeros(3), atol=0.1)
    # but hilbert is max
    assert np.isclose(t_max[1], true_t_max, atol=delta_t)
    assert idx_max[1] == int(true_t_max / 1000)
    true_max = np.sqrt(1 + 2 * 2 + 3 * 3)
    assert np.isclose(v_max[1], true_max, rtol=1e-3)
    assert t_max.shape == (2,) == v_max.shape
    assert idx_max.shape == (2, 1)
    assert norm_hilbert_amp.shape == (2, nb_s)


def test_get_fastest_size_fft():
    f_in = 10
    fastest_size_fft, freqs_mhz = get_fastest_size_fft(2048, f_in, padding_fact=1.99)
    assert fastest_size_fft == 4096
    assert freqs_mhz[0] == 0
    # delta freq
    assert freqs_mhz[1] == f_in / fastest_size_fft
    # Nyquist
    assert freqs_mhz[-1] == f_in / 2


def test_find_max_with_parabola_interp_3pt():
    def parabole(v_x):
        return -10 * v_x * v_x + 20 * v_x + 30

    n_sig = 29
    v_x = np.linspace(0, 3, n_sig)
    epsilon = 0.001
    v_y = parabole(v_x) + np.random.normal(0, epsilon, n_sig)
    idx = np.argmax(v_y)
    #print(idx, v_y[idx])
    x_max, y_max = find_max_with_parabola_interp_3pt(v_x, v_y, idx)
    print(x_max, y_max)
    assert np.isclose(x_max, 1.0, atol=epsilon * 10)
    assert np.isclose(y_max, 40.0, atol=epsilon * 10)


if __name__ == "__main__":
    test_get_peakamptime_norm_hilbert()
    tre.plt.show()
