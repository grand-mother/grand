"""
Handling a set of 3D traces
"""
from logging import getLogger
import copy

import numpy as np
import scipy.signal as ssig
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backend_bases import MouseButton

from grand.basis.du_network import DetectorUnitNetwork
import grand.basis.signal as gds


logger = getLogger(__name__)


def get_psd(trace, f_samp_mhz, nperseg=0):
    if nperseg == 0:
        nperseg = trace.shape[0] // 2

    freq, pxx_den = ssig.welch(
        trace, f_samp_mhz * 1e6, nperseg=nperseg, window="taylor", scaling="density"
    )
    return freq * 1e-6, pxx_den


class Handling3dTraces:
    """
    Handling a set of traces associated to one event observed on Detector Unit network

    Initialisation, with two methods:

       * init_traces()
       * optionally with init_network()

    Features:

        * some plots : trace , power spectrum, footprint, ...
        * compute time where the trace is maximun
        * compute time for each sample of trace
        * compute a common time for trace


    Public attributes:

        * name str: name of the set of trace
        * traces float(nb_du, 3, nb_sample): trace 3D
        * idx2idt int(nb_du): array of identifier of DU
        * t_start_ns float(nb_du): [ns] time of first sample of trace
        * t_samples float(nb_du, nb_dim, nb_sample): [ns]
        * f_samp_mhz float: [MHz] frequency sampling
        * idt2idx dict: for each identifier return the index in array
        * unit_trace str: string unit of trace
        * network object: content position network
    """

    def __init__(self, name="NotDefined"):
        logger.info(f"Create Handling3dTraces with name {name}")
        self.name = name
        self.info_shower = ""
        nb_du = 0
        nb_sample = 0
        self.nperseg = 512
        self.traces = np.zeros((nb_du, 3, nb_sample))
        self.idx2idt = range(nb_du)
        self.t_start_ns = np.zeros((nb_du), dtype=np.int64)
        self.t_samples = np.zeros((nb_du, nb_sample), dtype=np.float64)
        self.f_samp_mhz = 0.0
        self.idt2idx = {}
        self.unit_trace = "TBD"
        self.type_trace = ""
        self._d_axis_val = {
            "idx": ["0", "1", "2"],
            "port": ["1", "2", "3"],
            "cart": ["X", "Y", "Z"],
            "dir": ["SN", "EW", "UP"],
        }
        # blue for UP because the sky is blue
        # yellow for EW because sun is yellow
        #  and it rises in the west and sets in the east
        # k for black because the poles are white
        #  and the reverse of white (not visible on plot) is black
        self._color = ["k", "y", "b"]
        self.axis_name = self._d_axis_val["idx"]
        self.network = DetectorUnitNetwork()

    ### INTERNAL

    ### INIT/SETTER

    def init_traces(self, traces, du_id=None, t_start_ns=None, f_samp_mhz=2000):
        """

        :param traces: array traces 3D
        :type traces: float (nb_du, 3, nb sample)
        :param du_id:array identifier of DU
        :type du_id: int (nb_du,)
        :param t_start_ns: array time start of trace
        :type t_start_ns: int (nb_du,)
        :param f_samp_mhz: frequency sampling in MHz
        :type f_samp_mhz: float or array
        """
        assert isinstance(self.traces, np.ndarray)
        assert traces.ndim == 3
        assert traces.shape[1] == 3
        self.traces = traces
        if du_id is None:
            du_id = list(range(traces.shape[0]))
        if t_start_ns is None:
            t_start_ns = np.zeros(traces.shape[0], dtype=np.float32)
        self.idx2idt = du_id
        for idx, ident in enumerate(self.idx2idt):
            self.idt2idx[ident] = idx
        self.t_start_ns = t_start_ns
        if isinstance(f_samp_mhz, int) or isinstance(f_samp_mhz, float):
            self.f_samp_mhz = np.ones(len(du_id)) * f_samp_mhz
        else:
            self.f_samp_mhz = f_samp_mhz
        assert isinstance(self.t_start_ns, np.ndarray)
        assert traces.shape[0] == len(du_id)
        assert len(du_id) == t_start_ns.shape[0]
        self._define_t_samples()

    def init_network(self, du_pos):
        """

        :param du_pos:
        :type du_pos:
        """
        self.network.init_pos_id(du_pos, self.idx2idt)

    def set_unit_axis(self, str_unit="TBD", axis_name="idx", type_tr="Trace"):
        """

        :param str_unit:
        :type str_unit:
        :param axis_name:
        :type axis_name:
        :param type:
        :type type:
        """
        assert isinstance(str_unit, str)
        assert isinstance(axis_name, str)
        assert isinstance(type_tr, str)
        self.type_trace = type_tr
        self.unit_trace = str_unit
        self.axis_name = self._d_axis_val[axis_name]

    def set_periodogram(self, size):
        assert size > 0
        self.nperseg = size

    ### OPERATIONS

    def _define_t_samples(self):
        """
        Define time sample for the duration of the trace
        """
        if self.t_samples.size == 0:
            delta_ns = 1e3 / self.f_samp_mhz
            nb_sample = self.traces.shape[2]
            # to use numpy broadcast I need to transpose
            t_trace = (
                np.outer(
                    np.arange(0, nb_sample, dtype=np.float64),
                    delta_ns * np.ones(self.traces.shape[0]),
                )
                + self.t_start_ns
            )
            self.t_samples = t_trace.transpose()
            logger.info(f"shape t_samples =  {self.t_samples.shape}")

    def reduce_l_ident(self, l_idt):
        l_idx = [self.idt2idx[idt] for idt in l_idt]
        self.reduce_l_index(l_idx)

    def reduce_l_index(self, l_idx):
        print(self.idx2idt)
        print(type(self.idx2idt))
        print(type(l_idx))
        du_id = [self.idx2idt[idx] for idx in l_idx]
        self.idx2idt = du_id
        self.idt2idx = {}
        for idx, ident in enumerate(self.idx2idt):
            self.idt2idx[ident] = idx
        print(self.idx2idt)
        self.traces = self.traces[l_idx]
        self.t_start_ns = self.t_start_ns[l_idx]
        if self.t_samples.shape[0] > 0:
            self.t_samples = self.t_samples[l_idx]
        self.network = copy.deepcopy(self.network)
        self.network.reduce_l_index(l_idx)

    def reduce_nb_du(self, new_nb_du):
        """
        feature to reduce computation, for debugging

        :param new_nb_du: keep only new_nb_du first DU
        :type new_nb_du: int
        """
        assert new_nb_du > 0
        assert new_nb_du <= self.get_nb_du()
        self.idx2idt = self.idx2idt[:new_nb_du]
        self.traces = self.traces[:new_nb_du, :, :]
        self.t_start_ns = self.t_start_ns[:new_nb_du]
        if self.t_samples.shape[0] > 0:
            self.t_samples = self.t_samples[:new_nb_du, :, :]
        self.network.reduce_nb_du(new_nb_du)

    def downsize_sampling(self, fact):
        # self.traces = self.traces[:, :, ::fact]
        self.traces = ssig.decimate(self.traces, fact)
        self.f_samp_mhz /= fact
        self.t_samples = np.zeros((0, 0), dtype=np.float64)
        self._define_t_samples()
        self.nperseg = np.min(np.array([self.traces.shape[2] // 2, self.nperseg]))

    def remove_traces_low_signal(self, threshold):
        a_norm = np.max(np.max(np.abs(self.traces), axis=1), axis=1)
        l_idx_ok = []
        for idx in range(self.get_nb_du()):
            if a_norm[idx] >= threshold:
                l_idx_ok.append(idx)
        print(l_idx_ok)
        # l_idx_ok  = np.array(l_idx_ok)
        self.reduce_l_index(l_idx_ok)
        return l_idx_ok

    ### GETTER :
    def get_copy(self, new_traces=None, deepcopy=False):
        """Return a copy of current object where traces can be modify

        The type of copy is copy with reference, not a deepcopy
        https://stackoverflow.com/questions/3975376/why-updating-shallow-copy-dictionary-doesnt-update-original-dictionary/3975388#3975388

        if new_traces is :
          * None : object with same value
          * 0 : the return object has a traces with same shape but set to 0
          * np.array : the return object has new_traces as traces

        :param new_traces: if array must be have the same shape
        :type new_traces: array/None/0
        :return:
        """
        if deepcopy:
            my_copy = copy.deepcopy(self)
        else:
            my_copy = copy.copy(self)
        if new_traces is not None:
            if isinstance(new_traces, np.ndarray):
                assert self.traces.shape == new_traces.shape
            elif new_traces == 0:
                new_traces = np.zeros_like(self.traces)
            my_copy.traces = new_traces
            try:
                delattr(self, "t_max")
                delattr(self, "v_max")
            except:
                pass
        return my_copy

    def get_delta_t_ns(self):
        """
        Return sampling rate in ns
        """
        ret = 1e3 / self.f_samp_mhz
        return ret

    def get_max_abs(self):
        """
        Find absolute maximal value in trace for each detector

        :return:  array max of abs value
        :rtype: float (nb_du,)
        """
        return np.max(np.abs(self.traces), axis=(1, 2))

    def get_max_norm(self):
        """
        Return array of maximal of 3D norm in trace for each detector

        :return: array norm of traces
        :rtype: float (nb_du,)
        """
        # norm on 3D composant => axis=1
        # max on all norm => axis=1
        return np.max(np.linalg.norm(self.traces, axis=1), axis=1)

    def get_norm(self):
        """
        Return norm of traces for each time sample

        :return:  norm of traces for each time sample
        :rtype: float (nb_du, nb sample)
        """
        return np.linalg.norm(self.traces, axis=1)

    def get_tmax_vmax(self, hilbert=True, interpol="auto"):
        """
        Return time where norm of the amplitude of the Hilbert tranform  is max

        :param hilbert: True for Hilbert envelop else norm L2
        :type hilbert: bool
        :param interpol: keyword in no, auto, parab
        :type interpol: string
        :return: time of max and max
        :rtype: float(nb_du,) , float(nb_du,)
        """
        if hilbert:
            tmax, vmax, idx_max, tr_norm = gds.get_peakamptime_norm_hilbert(
                self.t_samples, self.traces
            )
        else:
            tr_norm = np.linalg.norm(self.traces, axis=1)
            idx_max = np.argmax(tr_norm, axis=1)
            idx_max = idx_max[:, np.newaxis]
            tmax = np.squeeze(np.take_along_axis(self.t_samples, idx_max, axis=1))
            vmax = np.squeeze(np.take_along_axis(tr_norm, idx_max, axis=1))
        if interpol == "no":
            self.t_max = tmax
            self.v_max = vmax
            return tmax, vmax
        if not interpol in ["parab", "auto"]:
            raise
        t_max = np.empty_like(tmax)
        v_max = np.empty_like(tmax)
        for idx in range(self.get_nb_du()):
            logger.debug(f"{idx} {self.idx2idt[idx]} {idx_max[idx]}")
            if interpol == "parab":
                t_max[idx], v_max[idx] = gds.find_max_with_parabola_interp_3pt(
                    self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
                )
            else:
                t_max[idx], v_max[idx] = gds.find_max_with_parabola_interp(
                    self.t_samples[idx], tr_norm[idx], int(idx_max[idx])
                )
            logger.debug(f"{t_max[idx]} ; {v_max[idx]}")
        self.t_max = t_max
        self.v_max = v_max
        return t_max, v_max

    def get_min_max_t_start(self):
        """
        :return: first and last time start
        :rtype: float, float
        """
        return self.t_start_ns.min(), self.t_start_ns.max()

    def get_nb_du(self):
        """
        :return: number of DU
        :rtype: int
        """
        return len(self.idx2idt)

    def get_size_trace(self):
        """
        :return: number of sample in trace
        :rtype: int
        """
        return self.traces.shape[2]

    def get_extended_traces(self):
        """
        compute and return traces extended to the entire duration of the event with common time

        :return: common time, extended traces
        :rtype: float (nb extended sample), float (nb_du, 3, nb extended sample)
        """
        size_tr = int(self.get_size_trace())
        t_min, t_max = self.get_min_max_t_start()
        delta = self.get_delta_t_ns()[0]
        nb_sample_mm = (t_max - t_min) / delta
        nb_sample = int(np.rint(nb_sample_mm) + size_tr)
        extended_traces = np.zeros((self.get_nb_du(), 3, nb_sample), dtype=self.traces.dtype)
        # don't use np.uint64 else int+ int =float ??
        i_beg = np.rint((self.t_start_ns - t_min) / delta).astype(np.uint32)
        for idx in range(self.get_nb_du()):
            extended_traces[idx, :, i_beg[idx] : i_beg[idx] + size_tr] = self.traces[idx]
        common_time = t_min + np.arange(nb_sample, dtype=np.float64) * delta
        return common_time, extended_traces

    ### PLOTS

    def plot_trace_idx(self, idx, to_draw="012"):  # pragma: no cover
        """
        Draw 3 traces associated to DU with index idx

        :param idx: index of DU to draw
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self._define_t_samples()
        plt.figure()
        s_title = f"{self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        s_title += f"\n$F_{{sampling}}$={self.f_samp_mhz[idx]}MHz"
        s_title += f"; {self.get_size_trace()} samples"
        plt.title(s_title)
        for idx_axis, axis in enumerate(self.axis_name):
            if str(idx_axis) in to_draw:
                m_sig = np.std(self.traces[idx, idx_axis, :100])
                plt.plot(
                    self.t_samples[idx],
                    self.traces[idx, idx_axis],
                    self._color[idx_axis],
                    label=axis + f", $\sigma=${m_sig:.2e}",
                )
        if hasattr(self, "t_max"):
            plt.plot(
                self.t_max[idx],
                self.v_max[idx],
                "d",
                label=f"max {self.v_max[idx]:e}",
            )
        plt.ylabel(f"{self.unit_trace}")
        plt.xlabel(f"ns\n{self.name}")
        plt.grid()
        plt.legend()

    def plot_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """
        Draw 3 traces associated to DU idx2idt

        :param idx: index of DU to draw
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_trace_idx(self.idt2idx[du_id], to_draw)

    def plot_ps_trace_idx(self, idx, to_draw="012"):  # pragma: no cover
        """
        Draw power spectrum for 3 traces associated to DU at index idx

        :param idx: index of trace
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self._define_t_samples()
        plt.figure()
        for idx_axis, axis in enumerate(self.axis_name):
            if str(idx_axis) in to_draw:
                if True:
                    freq, pxx_den = ssig.welch(
                        self.traces[idx, idx_axis],
                        self.f_samp_mhz[idx] * 1e6,
                        nperseg=self.nperseg,
                        window="bartlett",
                        scaling="density",
                    )
                else:
                    freq, pxx_den = get_psd(self.traces[idx, idx_axis], self.f_samp_mhz)
                plt.semilogy(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
                # plt.plot(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
        m_title = f"Power spectrum density of {self.type_trace}, DU {self.idx2idt[idx]} (idx={idx})"
        m_title += f"\nPeriodogram have {self.nperseg} samples, delta freq {freq[1]*1e-6:.2f}MHz"
        plt.title(m_title)
        plt.ylabel(rf"({self.unit_trace})$^2$/Hz")
        plt.xlabel(f"MHz\n{self.name}")
        plt.xlim([0, 400])
        plt.grid()
        plt.legend()
        self.welch_freq = freq
        self.welch_pxx_den = pxx_den

    def plot_ps_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """
        Draw power spectrum for 3 traces associated to DU idx2idt

        :param idx2idt: DU identifier
        :type idx2idt: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_ps_trace_idx(self.idt2idx[du_id], to_draw)

    def plot_all_traces_as_image(self):  # pragma: no cover
        """
        Interactive image double click open traces associated
        """
        norm = self.get_norm()
        _ = plt.figure()
        # fig.canvas.manager.set_window_title(f"{self.name}")
        plt.title(f"Norm of all traces {self.type_trace} in event")
        col_log = colors.LogNorm(clip=False)
        im_traces = plt.imshow(norm, cmap="Blues", norm=col_log)
        plt.colorbar(im_traces)
        plt.xlabel(f"Index sample\nFile: {self.name}")
        plt.ylabel("Index DU")

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                idx = int(event.ydata + 0.5)
                self.plot_trace_idx(idx)
                self.plot_ps_trace_idx(idx)
                plt.show()

        plt.connect("button_press_event", on_click)

    def plot_histo_t_start(self):  # pragma: no cover
        """
        Histogram of time start
        """
        plt.figure()
        plt.title(rf"{self.name}\nTime start histogram")
        plt.hist(self.t_start_ns)
        plt.xlabel("ns")
        plt.grid()

    def plot_footprint_4d_max(self):  # pragma: no cover
        """
        Plot time max and max value by component
        """
        v_max = np.max(np.abs(self.traces), axis=2)
        self.network.plot_footprint_4d(self, v_max, "3D", unit=self.unit_trace)

    def plot_footprint_val_max(self):  # pragma: no cover
        """
        Plot footprint max value
        """
        self.network.plot_footprint_1d(
            self.get_max_norm(), f"Max ||{self.type_trace}||", self, unit=self.unit_trace
        )

    def plot_footprint_time_max(self):  # pragma: no cover
        """
        Plot footprint time associated to max value
        """
        tmax, _ = self.get_tmax_vmax(False)
        self.network.plot_footprint_1d(tmax, "Time of max value", self, scale="lin", unit="ns")

    def plot_footprint_time_slider(self):  # pragma: no cover
        """
        Plot footprint max value
        """
        if self.network:
            a_time, a_values = self.get_extended_traces()
            self.network.plot_footprint_time(a_time, a_values, "Max value")
        else:
            logger.error("DU network isn't defined, can't plot footprint")
