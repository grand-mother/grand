"""
Handling a set of 3D traces
"""
from logging import getLogger

import numpy as np
import scipy.signal as ssig
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backend_bases import MouseButton

from grand.basis.du_network import DetectorUnitNetwork
import grand.num.signal as gsig


logger = getLogger(__name__)


class Handling3dTracesOfEvent:
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
        * du_id int(nb_du): array of identifier of DU
        * t_start_ns float(nb_du): [ns] time of first sample of trace
        * t_samples float(nb_du, nb_dim, nb_sample): [ns]
        * f_samp_mhz float: [MHz] frequency sampling
        * d_idxdu dict: for each identifier return the index in array
        * unit_trace str: string unit of trace
        * network object: content position network
    """

    def __init__(self, name="NotDefined"):
        logger.info(f"Create Handling3dTracesOfEvent with name {name}")
        self.name = name
        nb_du = 0
        nb_sample = 0
        self.traces = np.zeros((nb_du, 3, nb_sample))
        self.du_id = np.arange(nb_du)
        self.t_start_ns = np.zeros((nb_du), dtype=np.int64)
        self.t_samples = np.zeros((nb_du, nb_sample), dtype=np.float64)
        self.f_samp_mhz = 0.0
        self.d_idxdu = {}
        self.unit_trace = "TBD"
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
        self.network = DetectorUnitNetwork(self.name)

    ### INTERNAL

    ### INIT/SETTER

    def init_traces(self, traces, du_id, t_start_ns, f_samp_mhz):
        """

        :param traces: array traces 3D
        :type traces: float (nb_du, 3, nb sample)
        :param du_id: array identifier of DU
        :type du_id: int (nb_du,)
        :param t_start_ns: array time start of trace
        :type t_start_ns: int (nb_du,)
        :param f_samp_mhz: franquency sampling in MHz
        :type f_samp_mhz: float
        """
        self.traces = traces
        self.du_id = du_id
        for idx, ident in enumerate(self.du_id):
            self.d_idxdu[ident] = idx
        self.t_start_ns = t_start_ns
        self.f_samp_mhz = f_samp_mhz
        assert isinstance(self.traces, np.ndarray)
        assert isinstance(self.t_start_ns, np.ndarray)
        assert traces.shape[1] == 3
        assert traces.shape[0] == du_id.shape[0]
        assert du_id.shape[0] == t_start_ns.shape[0]
        self._define_t_samples()

    def init_network(self, du_pos, du_id):
        """

        :param du_pos:
        :type du_pos:
        :param du_id:
        :type du_id:
        """
        self.network.init_pos_id(du_pos, du_id)

    def set_unit_axis(self, str_unit="TBD", axis_name="idx"):
        """

        :param str_unit:
        :type str_unit:
        :param axis_name:
        :type axis_name:
        """
        assert isinstance(str_unit, str)
        self.unit_trace = str_unit
        self.axis_name = self._d_axis_val[axis_name]

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

    def reduce_nb_du(self, new_nb_du):
        """
        feature to reduce computation, for debugging

        :param new_nb_du: keep only new_nb_du first DU
        :type new_nb_du: int
        """
        assert new_nb_du > 0
        assert new_nb_du <= self.get_nb_du()
        self.du_id = self.du_id[:new_nb_du]
        self.traces = self.traces[:new_nb_du, :, :]
        self.t_start_ns = self.t_start_ns[:new_nb_du]
        if self.t_samples.shape[0] > 0:
            self.t_samples = self.t_samples[:new_nb_du, :, :]
        self.network.reduce_nb_du(new_nb_du)

    ### GETTER :

    def delta_t_ns(self):
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

    def get_tmax_vmax(self):
        """
        Return time where norm of the amplitude of the Hilbert tranform  is max

        :return:  time of max and max
        :rtype: float(nb_du,) , float(nb_du,)
        """
        tmax, vmax, _, _ = gsig.get_peakamptime_norm_hilbert(self.t_samples, self.traces)
        return tmax, vmax

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
        return self.du_id.shape[0]

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
        delta = self.delta_t_ns()
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
        plt.title(f"Trace of DU {self.du_id[idx]} (idx={idx})")
        for idx_axis, axis in enumerate(self.axis_name):
            if str(idx_axis) in to_draw:
                plt.plot(
                    self.t_samples[idx],
                    self.traces[idx, idx_axis],
                    self._color[idx_axis],
                    label=axis,
                )
        plt.ylabel(f"{self.unit_trace}")
        plt.xlabel(f"ns\nFile: {self.name}")
        plt.grid()
        plt.legend()

    def plot_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """
        Draw 3 traces associated to DU du_id

        :param idx: index of DU to draw
        :type idx: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_trace_idx(self.d_idxdu[du_id], to_draw)

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
        noverlap = 0
        plt.title(f"Power spectrum of DU {self.du_id[idx]} (idx={idx})")
        for idx_axis, axis in enumerate(self.axis_name):
            if str(idx_axis) in to_draw:
                freq, pxx_den = ssig.welch(
                    self.traces[idx, idx_axis],
                    self.f_samp_mhz * 1e6,
                    window="taylor",
                    noverlap=noverlap,
                    scaling="spectrum",
                )
                plt.semilogy(freq * 1e-6, pxx_den, self._color[idx_axis], label=axis)
        plt.ylabel(f"({self.unit_trace})^2")
        plt.xlabel(f"MHz\nFile: {self.name}")
        plt.xlim([0, 300])
        plt.grid()
        plt.legend()

    def plot_ps_trace_du(self, du_id, to_draw="012"):  # pragma: no cover
        """
        Draw power spectrum for 3 traces associated to DU du_id

        :param du_id: DU identifier
        :type du_id: int
        :param to_draw: select components to draw
        :type to_draw: enum str ["0", "1", "2"] not exclusive
        """
        self.plot_ps_trace_idx(self.d_idxdu[du_id], to_draw)

    def plot_all_traces_as_image(self):  # pragma: no cover
        """
        Interactive image double click open traces associated
        """
        norm = self.get_norm()
        _ = plt.figure()
        # fig.canvas.manager.set_window_title(f"{self.name}")
        plt.title(f"Norm of all traces in event")
        col_log = colors.LogNorm(clip=False)
        im_traces = plt.imshow(norm, cmap="Blues", norm=col_log)
        plt.colorbar(im_traces)
        plt.xlabel(f"Index sample\nFile: {self.name}")
        plt.ylabel("Index DU")

        def on_click(event):
            if event.button is MouseButton.LEFT and event.dblclick:
                self.plot_trace_idx(int(event.ydata + 0.5))
                plt.show()

        plt.connect("button_press_event", on_click)

    def plot_histo_t_start(self):  # pragma: no cover
        """
        Histogram of time start
        """
        plt.figure()
        plt.title(f"{self.name}\nTime start histogram")
        plt.hist(self.t_start_ns)
        plt.xlabel("ns")
        plt.grid()

    def plot_footprint_4d(self):  # pragma: no cover
        """
        Plot time max and max value by component
        """
        self.network.plot_footprint_4d(self, "3D")

    def plot_footprint_val_max(self):  # pragma: no cover
        """
        Plot footprint max value
        """
        self.network.plot_footprint_1d(self.get_max_norm(), "Max ||Efield||", self)

    def plot_footprint_time_max(self):  # pragma: no cover
        """
        Plot footprint time associated to max value
        """
        tmax, _ = self.get_tmax_vmax()
        self.network.plot_footprint_1d(tmax, "Time of max value", self, scale="lin")
