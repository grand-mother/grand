"""
Manage ROOT file of type Efield simulation, voltage ADCevent is experimental 
"""

import os.path
from logging import getLogger

import numpy as np
import ROOT

import grand.io.root_trees as groot
from grand.basis.traces_event import Handling3dTracesOfEvent

logger = getLogger(__name__)


def get_ttree_in_file(f_root):
    """
    Return all TTree name in ROOT file

    :param f_root:
    :type f_root:
    """
    tfile = ROOT.TFile.Open(f_root)
    l_ttree = tfile.GetListOfKeys()
    l_name_ttree = [ttree.GetName() for ttree in l_ttree]
    return l_name_ttree


def check_ttree_in_file(f_root, ttree_name):
    """
    Return True if f_root contents ttree_name

    :param f_root:
    :type f_root:
    :param ttree_name:
    :type ttree_name:
    """
    return ttree_name in get_ttree_in_file(f_root)


class _FileEventBase:
    """
    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary


    Public attributes:

        * tt_event object: object ROOT_tree of type TTree Event
        * f_name str: path/name to ROOT file
        * l_events list : list of tuple (event, run)
        * traces float(N,3,S): 3D trace of N detecttor unit with S samples
        * idx_event int:  current object contents data of event at index idx_event in l_events
        * delta_t_ns float: [ns] time sampling
        * evt_nb int: given by l_events list(idx_event)
        * run_nb int: given by l_events list(idx_event)
        * nb_du int: number of DU in current object
        * du_pos float(N,3): cartesian position of detector unit
    """

    def __init__(self, tt_event):
        """
        Constructor
        """
        self.tt_event = tt_event
        self.l_events = self.tt_event.get_list_of_events()
        self.traces = np.empty((0, 3, 0), dtype=np.float32)
        # number detector in event
        self.idx_event = -1
        self.delta_t_ns = 0.5
        self.evt_nb = -1
        self.run_nb = -1
        self.nb_du = 0
        self.du_pos = np.empty((self.nb_du, 3), dtype=np.float32)
        self.f_name = ""

    def load_event_idx(self, idx):
        """
        Load event/run with index idx in list return by get_list_of_events()

        :param idx:
        """
        if self.idx_event == idx:
            return
        self.idx_event = idx
        self.evt_nb = self.l_events[idx][0]
        self.run_nb = self.l_events[idx][1]
        self._load_event_identifier(self.evt_nb, self.run_nb)

    def load_next_event(self):
        """
        Load next event, return False at the end of list else True
        """
        idx = self.idx_event + 1
        if idx >= self.get_nb_events():
            logger.warning(f"No more event")
            return False
        return self.load_event_idx(idx)

    def _load_event_identifier(self, evt_nb, run_nb):
        """
        Load traces/pos of event/run evt_nb/run_nb
        :param evt_nb:
        :param run_nb:
        """
        logger.info(f"load event: {evt_nb} of run  {run_nb}")
        # efield
        self.tt_event.get_event(evt_nb, run_nb)
        self.nb_du = self.tt_event.du_count
        trace_size = len(self.tt_event.trace_x[0])
        if trace_size != self.traces.shape[2]:
            self.traces = np.empty((self.nb_du, 3, trace_size), dtype=np.float32)
            logger.info(f"resize numpy array trace to {self.traces.shape}")
        self.traces[:, 0, :] = np.array(self.tt_event.trace_x, dtype=np.float32)
        self.traces[:, 1, :] = np.array(self.tt_event.trace_y, dtype=np.float32)
        self.traces[:, 2, :] = np.array(self.tt_event.trace_z, dtype=np.float32)
        self.du_pos = np.empty((self.nb_du, 3), dtype=np.float32)
        self.du_pos[:, 0] = np.array(self.tt_event.pos_x, dtype=np.float32)
        self.du_pos[:, 1] = np.array(self.tt_event.pos_y, dtype=np.float32)
        self.du_pos[:, 2] = np.array(self.tt_event.pos_z, dtype=np.float32)

    def get_nb_du(self):
        """
        Return number of du in event
        """
        return self.tt_event.du_count

    def get_nb_events(self):
        """
        Return number of event in file
        """
        return len(self.l_events)

    def get_size_trace(self):
        """
        Return number of sample in trace
        """
        return self.traces.shape[2]

    def get_sampling_freq_mhz(self):
        """
        Return sampling frequency in MHz
        """
        return 1e3 / self.delta_t_ns

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent Handling3dTracesOfEvent
        """
        o_tevent = Handling3dTracesOfEvent(f"{self.f_name} evt={self.evt_nb} run={self.run_nb}")
        # TODO: difference between du_id for all network and for specific event ?
        du_id = np.array(self.tt_event.du_id)
        o_tevent.init_traces(
            self.traces,
            du_id,
            np.array(self.tt_event.du_nanoseconds),
            self.get_sampling_freq_mhz(),
        )
        o_tevent.init_network(self.du_pos, du_id)
        return o_tevent


class FileSimuEfield(_FileEventBase):
    """
    File simulation of air shower with 5 TTree

      * teventefield
      * teventshowersimdata
      * teventshowerzhaires
      * trun
      * trunefieldsimdata

    Goals of the class:

      * synchronize each TTree on same event/run
      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributes:

      * same as _FileEventBase class
      * tt_efield object EfieldEventTree
      * tt_shower object ShowerEventSimdataTree
      * tt_run object RunTree
      * tt_run_sim object EfieldRunSimdataTree

    """

    def __init__(self, f_name, check=True):
        """
        Constructor
        """
        name_ttree = "teventefield"
        if check:
            if not os.path.exists(f_name):
                logger.error(f"File {f_name} doesn't exist.")
                raise FileNotFoundError
            if not check_ttree_in_file(f_name, name_ttree):
                logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
                raise AssertionError
        logger.info(f"Events  in file {f_name}")
        event = groot.EfieldEventTree(f_name)
        super().__init__(event)
        self.tt_efield = self.tt_event
        self.tt_shower = groot.ShowerEventSimdataTree(f_name)
        self.tt_run = groot.RunTree(f_name)
        self.tt_run_sim = groot.EfieldRunSimdataTree(f_name)
        self.load_event_idx(0)
        self.f_name = f_name

    def _load_event_identifier(self, evt_nb, run_nb):
        """
        synchronize all event TTree with method get_event() in file.

        :param evt_nb:
        :param run_nb:
        """
        super()._load_event_identifier(evt_nb, run_nb)
        # synchronize showerTree on same evt, run
        logger.info(f"load tt_shower: {evt_nb} of run  {run_nb}")
        self.tt_shower.get_event(evt_nb, run_nb)
        # synchronize runTree on same run
        self.tt_run.get_run(run_nb)
        self.tt_shower.site_long_lat = np.array([self.tt_run.site_long, self.tt_run.site_lat])
        # synchronize EfieldRunSimdataTree on same run
        self.tt_run_sim.get_run(run_nb)
        self.delta_t_ns = self.tt_run_sim.t_bin_size

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis("$\mu$V/m", "cart")
        return o_tevent


class FileVoltageEvent(_FileEventBase):
    """
    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributs:

      * same as _FileEventBase class
    """

    def __init__(self, f_name, check=True):
        """
        Constructor
        """
        name_ttree = "teventvoltage"
        if check:
            if not os.path.exists(f_name):
                logger.error(f"File {f_name} doesn't exist.")
                raise FileNotFoundError
            if not check_ttree_in_file(f_name, name_ttree):
                logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
                raise AssertionError
        logger.info(f"Events  in file {f_name}")
        event = groot.VoltageEventTree(f_name)
        super().__init__(event)
        self.load_event_idx(0)
        self.f_name = f_name

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis("$\mu$V", "dir")
        return o_tevent



    
def get_file_event(f_name):
    """
    Factory for ROOT event file, return an instance of FileSimuEfield or FileVoltageEvent
    """
    if not os.path.exists(f_name):
        print('File does not exist')
        logger.error(f"File {f_name} doesn't exist.")
        raise FileNotFoundError
    else:
        trees_list = get_ttree_in_file(f_name)
        if "teventefield" in trees_list:          # File with Efield info as input
            return FileSimuEfield(f_name, False)
        elif "teventvoltage" in trees_list:       # File with voltage info as input
            return FileVoltageEvent(f_name, False)
        else:
            logger.error(f"File {f_name} doesn't content TTree teventefield or teventvoltage. It contains {trees_list}.")
            raise AssertionError

        
class FileADCeventProto(_FileEventBase):
    """

    .. warning: Prototype for ADC event, work in progress


    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributes:

      * same as _FileEventBase class
      * all_traces array(nb_all_event_in_file, nb_sample)
    """

    def __init__(self, f_name):
        """
        Constructor
        """
        self.warning = True
        name_ttree = "teventadc"
        if not os.path.exists(f_name):
            logger.error(f"File {f_name} doesn't exist.")
            raise FileNotFoundError
        if not check_ttree_in_file(f_name, name_ttree):
            logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
            raise AssertionError
        logger.info(f"Events  in file {f_name}")
        event = groot.ADCEventTree(f_name)
        super().__init__(event)
        self.load_event_idx(0)
        self.f_name = f_name
        nb_sample = 100
        nb_all_event_in_file = 0
        self.all_traces = np.array((nb_all_event_in_file, nb_sample), dtype=np.float32)

    def get_obj_handlingtracesofevent_all(self):
        """
        Return a traces container IO independent
        """
        self._load_all_events()
        o_tevent = Handling3dTracesOfEvent(f"{self.f_name} all events")
        # TODO: difference between du_id for all network and for specific event ?
        du_id = self.tt_event.du_id * np.ones(self.get_nb_events())
        o_tevent.init_traces(
            self.all_traces,
            du_id,
            np.zeros(self.get_nb_events()),
            self.get_sampling_freq_mhz(),
        )
        o_tevent.set_unit_axis("ADU", "idx")
        return o_tevent

    def _load_all_events(self):
        """
        Internal method to read first trace for all event

        :param self:
        :type self:
        """
        self.all_traces = np.zeros(
            (self.get_nb_events(), 3, self.get_size_trace()), dtype=np.float32
        )
        for idx in range(self.get_nb_events()):
            self.load_event_idx(idx)
            self.all_traces[idx] = self.traces[0]

    def _load_event_identifier(self, evt_nb, run_nb):
        """
        Load traces/pos of event/run evt_nb/run_nb

        :param evt_nb:
        :param run_nb:
        """
        logger.info(f"load event: {evt_nb} of run  {run_nb}")
        # efield
        self.tt_event.get_event(evt_nb, run_nb)
        self.nb_du = self.tt_event.du_count
        trace_size = len(self.tt_event.trace_0[0])
        if trace_size != self.traces.shape[2]:
            self.traces = np.zeros((self.nb_du, 3, trace_size), dtype=np.float32)
            logger.info(f"resize numpy array trace to {self.traces.shape}")
        self.traces[:, 0, :] = np.array(self.tt_event.trace_0, dtype=np.float32)
        try:
            self.traces[:, 1, :] = np.array(self.tt_event.trace_1, dtype=np.float32)
        except:
            if self.warning:
                logger.warning(f"trace_1 isn't filled")
        try:
            self.traces[:, 2, :] = np.array(self.tt_event.trace_2, dtype=np.float32)
        except:
            if self.warning:
                logger.warning(f"trace_2 isn't filled")
        self.warning = False
        self.du_pos = np.empty((self.nb_du, 3), dtype=np.float32)
        self.du_pos[:, 0] = np.array(self.tt_event.gps_long, dtype=np.float32)
        self.du_pos[:, 1] = np.array(self.tt_event.gps_lat, dtype=np.float32)
        self.du_pos[:, 2] = np.array(self.tt_event.gps_alt, dtype=np.float32)
