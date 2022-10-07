from logging import getLogger

import numpy as np
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
        self.traces = None
        self.du_id = None
        self.t_start_ns = None
        self.t_samples = False
        self.f_samp_mhz = None
        self.unit_trace = "TBD"
        self.network = DetectorUnitNetwork(self.name)

    ### INTERNAL

    def _aa(self):
        pass

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
        if not isinstance(self.t_samples, np.ndarray):
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

    def reduce_nb_du(self, new_nb_du):
        """
        feature to reduce computation and debugging
        :param new_nb_du:
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

    def get_max_abs(self):
        """
        find absolute maximal value in trace for each detector
        :param self:
        """
        return np.max(np.abs(self.traces), axis=(1, 2))

    def get_max_norm(self):
        """
        find norm maximal value in trace for each detector
        :param self:
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
        return self.du_id.shape[2]

    ### PLOTS

    def plot_trace_idx(self, idx, to_draw="xyz"):
        self.define_t_samples()
        plt.figure()
        plt.title(f"{self.name}\nTrace of DU {self.du_id[idx]} (idx={idx}) ")
        if "x" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 0], label="x")
        if "y" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 1], label="y")
        if "z" in to_draw:
            plt.plot(self.t_samples[idx], self.traces[idx, 2], label="z")
        plt.ylabel(f"[{self.unit_trace}]")
        plt.xlabel("[ns]")
        plt.grid()
        plt.legend()

    def plot_histo_t_start(self):
        plt.figure()
        plt.title(f"{self.name}\nTime start histogram")
        plt.hist(self.t_start_ns)
        plt.xlabel("[ns]")
        plt.grid()
