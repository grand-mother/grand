from logging import getLogger

import numpy as np
import scipy.signal as ssig
import matplotlib.pyplot as plt

from grand.basis.du_network import DetectorUnitNetwork


logger = getLogger(__name__)


class HandlingTracesOfEvent:
    """
    Handling a set of traces associated to one event observed on DetectorUnit network
    """

    def __init__(self, name="NotDefined"):
        logger.info(f"Create HandlingTracesOfEvent with name {name}")
        self.name = name
        nb_du = 0
        nb_dim = 3
        nb_sample = 0
        self.traces = np.zeros((nb_du, nb_dim, nb_sample))
        self.du_id = np.arange(nb_du)
        self.t_start_ns = np.zeros((nb_du), dtype=np.int64)
        self.t_samples = np.zeros((nb_du), dtype=np.float64)
        self.f_samp_mhz = 0
        self.unit_trace = "TBD"
        self.network = DetectorUnitNetwork(self.name)

    ### INTERNAL

    ### INIT/SETTER

    def init_traces(self, traces, du_id, t_start_ns, f_samp_mhz):
        self.traces = traces
        self.du_id = du_id
        self.t_start_ns = t_start_ns
        self.f_samp_mhz = f_samp_mhz
        assert isinstance(self.traces, np.ndarray)
        assert isinstance(self.t_start_ns, np.ndarray)
        assert traces.shape[1] == 3
        assert traces.shape[0] == du_id.shape[0] == t_start_ns.shape[0]

    def init_network(self, du_pos, du_id):
        self.network.init_pos_id(du_pos, du_id)

    def set_unit_trace(self, str_unit):
        assert isinstance(str_unit, str)
        self.unit_trace = str_unit

    ### OPERATIONS

    def define_t_samples(self):
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
        @param new_nb_du:
        """
        assert new_nb_du > 0
        assert new_nb_du <= self.get_nb_du()
        self.du_id = self.du_id[:new_nb_du]
        self.traces = self.traces[:new_nb_du, :, :]
        self.t_start_ns = self.t_start_ns[:new_nb_du]
        if isinstance(self.t_samples, np.ndarray):
            self.t_samples = self.t_samples[:new_nb_du, :, :]
        self.network.reduce_nb_du(new_nb_du)

    ### GETTER :

    def delta_t_ns(self):
        ret = 1.0 / (self.f_samp_mhz / 1e3)
        return ret

    def get_max_abs(self):
        """
        find absolute maximal value in trace for each detector
        @param self:
        """
        return np.max(np.abs(self.traces), axis=(1, 2))

    def get_max_norm(self):
        """
        find norm maximal value in trace for each detector
        @param self:
        """
        # norm on 3D composant => axis=1
        # max on all norm => axis=1
        return np.max(np.linalg.norm(self.traces, axis=1), axis=1)

    def get_norm(self):
        return np.linalg.norm(self.traces, axis=1)

    def get_tmax_vmax(self, method="efield"):
        """
        tmax vmax are input data for reconstruction

        method="efield"
           * compute norm of Efield and find max value and time associated

        method="du" (volt/adc)
           * to deal with oscillation take the hilbert's envelope of the norm
           * find max value and time associated, may be with interpolation with law sampling
        """
        pass

    def get_min_max_t_start(self):
        return self.t_start_ns.min(), self.t_start_ns.max()

    def get_nb_du(self):
        return self.du_id.shape[0]

    def get_size_trace(self):
        return self.traces.shape[2]

    def get_common_time_trace(self):
        size_tr = int(self.get_size_trace())
        t_min, t_max = self.get_min_max_t_start()
        delta = self.delta_t_ns()
        nb_sample_mm = (t_max - t_min) / delta
        nb_sample = int(np.rint(nb_sample_mm) + size_tr)
        new_traces = np.zeros((self.get_nb_du(), 3, nb_sample), dtype=self.traces.dtype)
        # don't use np.uint64 else int+ int =float ??
        i_beg = np.rint((self.t_start_ns - t_min) / delta).astype(np.uint32)
        for idx in range(self.get_nb_du()):
            new_traces[idx, :, i_beg[idx] : i_beg[idx] + size_tr] = self.traces[idx]
        new_time = t_min + np.arange(nb_sample, dtype=np.float64) * delta
        return new_time, new_traces

    ### PLOT

    def plot_trace_idx(self, idx, to_draw="xyz"):
        self.define_t_samples()
        plt.figure()
        plt.title(f"Trace of DU {self.du_id[idx]} (idx={idx})")
        if "x" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 0], label="x")
        if "y" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 1], label="y")
        if "z" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 2], label="z")
        plt.ylabel(f"[{self.unit_trace}]")
        plt.xlabel(f"[ns]\nFile: {self.name}")
        plt.grid()
        plt.legend()

    def plot_psd_trace_idx(self, idx, to_draw="xyz"):
        self.define_t_samples()
        plt.figure()
        noverlap = 10
        plt.title(f"PSD trace of DU {self.du_id[idx]} (idx={idx})")
        if "x" in to_draw:
            f, Pxx_den = ssig.welch(self.traces[idx, 0], self.f_samp_mhz * 1e6, noverlap=noverlap)
            plt.plot(f * 1e-6, Pxx_den, label="x")
        if "y" in to_draw:
            f, Pxx_den = ssig.welch(self.traces[idx, 1], self.f_samp_mhz * 1e6, noverlap=noverlap)
            plt.plot(f * 1e-6, Pxx_den, label="y")
        if "z" in to_draw:
            f, Pxx_den = ssig.welch(self.traces[idx, 2], self.f_samp_mhz * 1e6, noverlap=noverlap)
            plt.plot(f * 1e-6, Pxx_den, label="z")
        plt.ylabel(f"[??]")
        plt.xlabel(f"[MHz]\nFile: {self.name}")
        plt.xlim([0, 300])
        plt.grid()
        plt.legend()

    def plot_all_traces_as_image(self):  # pragma: no cover
        import matplotlib.colors as colors
        from matplotlib.backend_bases import MouseButton

        #
        norm = self.get_norm()
        fig = plt.figure()
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
        plt.figure()
        plt.title(f"{self.name}\nTime start histogram")
        plt.hist(self.t_start_ns)
        plt.xlabel("[ns]")
        plt.grid()
