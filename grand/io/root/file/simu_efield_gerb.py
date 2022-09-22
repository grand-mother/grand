"""

"""
import os.path
from logging import getLogger

import numpy as np

from grand.io.root.event.efield import EfieldEventTree
from grand.io.root.event.shower import ShowerEventSimdataTree

logger = getLogger(__name__)


class FileSimuEfield:
    """
    File simulation of gerb from ZHAireS is a root file with 5 TTree
      * teventefield
      * teventshowersimdata
      * teventshowerzhaires
      * trun
      * trunefieldsimdata

    Goals of the class:
      * synchronize each TTree on same event
      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary
    """

    def __init__(self, f_name):
        """
        Constructor
        """
        if not os.path.exists(f_name):
            raise
        self.tt_efield = EfieldEventTree(f_name)
        self.tt_shower = ShowerEventSimdataTree(f_name)
        self.l_events = self.tt_efield.get_list_of_events()
        logger.info(f"Events  in file {f_name} :\n{self.l_events}")
        self.traces = np.empty((0, 3, 0), dtype=np.float32)
        # number detector in event
        self.idx_event = -1
        self.load_event_idx(0)

    def load_event_idx(self, idx):
        if self.idx_event == idx:
            return
        self.idx_event = idx
        evt_nb = self.l_events[idx][0]
        run_nb = self.l_events[idx][1]
        self._load_event_identifier(evt_nb, run_nb)

    def _load_event_identifier(self, evt_nb, run_nb):
        """
        synchronize all event TTree with method get_event() in file.

        :param evt_nb:
        :param run_nb:
        """
        logger.info(f"load event: {evt_nb} of run  {run_nb}")
        # efield
        self.tt_efield.get_event(evt_nb, run_nb)
        self.nb_du = self.tt_efield.du_count
        trace_size = len(self.tt_efield.trace_x[0])
        if trace_size != self.traces.shape[2]:
            self.traces = np.empty((self.nb_du, 3, trace_size), dtype=np.float32)
            logger.info(f"resize numpy array trace to {self.traces.shape}")
        self.traces[:, 0, :] = np.array(self.tt_efield.trace_x, dtype=np.float32)
        self.traces[:, 1, :] = np.array(self.tt_efield.trace_y, dtype=np.float32)
        self.traces[:, 2, :] = np.array(self.tt_efield.trace_z, dtype=np.float32)
        self.du_pos = np.empty((self.nb_du, 3), dtype=np.float32)
        self.du_pos[:, 0] = np.array(self.tt_efield.pos_x, dtype=np.float32)
        self.du_pos[:, 1] = np.array(self.tt_efield.pos_y, dtype=np.float32)
        self.du_pos[:, 2] = np.array(self.tt_efield.pos_z, dtype=np.float32)
        # shower
        logger.info(f"load tt_shower: {evt_nb} of run  {run_nb}")
        self.tt_shower.get_event(evt_nb, run_nb)

    def get_nb_du(self):
        return self.traces.shape[0]

    def get_nb_events(self):
        return len(self.l_events)

    def get_size_trace(self):
        return self.traces.shape[2]

    def get_sampling_freq_mhz(self):
        # TODO: t_bin_size in EfieldRunSimdataTree class
        delta_t_ns = 0.5
        a_freq = (1e3 / delta_t_ns) * np.ones(self.get_nb_du(), dtype=np.float32)
        return a_freq

    def get_time_trace_ns(self):
        # TODO: check management of second , HYPOTHESIS second must bee all the same !!!!
        a_delta_t_ns = 1e3 / self.get_sampling_freq_mhz()
        nb_sample = self.traces.shape[2]
        # to use numpy broadcast I need to transpose
        t_trace = np.outer(np.arange(0, nb_sample, dtype=np.float64), a_delta_t_ns) + np.array(
            self.tt_efield.du_nanoseconds
        )
        return t_trace.transpose()
