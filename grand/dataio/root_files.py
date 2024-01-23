"""
Manage ROOT file of type Efield simulation, voltage
"""

import os.path
from logging import getLogger
from typing import Optional

import numpy as np
import ROOT

import grand.dataio.root_trees as groot
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
        * traces float(N,3,S): 3D trace of N detector unit with S samples
        * idx_event int:  current object contents data of event at index idx_event in l_events
        * delta_t_ns float: [ns] time sampling
        * event_number int: given by l_events list(idx_event)
        * run_number int: given by l_events list(idx_event)
        * du_count int: number of DU in current object
        * du_pos float(N,3): cartesian position of detector unit
    """

    def __init__(self, tt_event):
        """
        Constructor
        """
        self.run_number: Optional[None, int] = None
        self.event_number: Optional[None, int] = None
        self.traces: Optional[None, np.ndarray] = None
        self.sig_size: Optional[None, int] = None
        self.t_bin_size: Optional[float] = 0.5
        self.du_id: Optional[None, np, ndarray, list] = None
        self.du_count: Optional[None, int] = None
        self.du_xyz: Optional[None, np.ndarray] = None
        self.f_name: Optional[str] = ""
        self.tag: Optional[str] = ""

        self.tt_event = tt_event
        self.l_events = self.tt_event.get_list_of_events()
        self.traces = np.empty((0, 3, 0), dtype=np.float32)
        self.idx_event = -1
        # self.delta_t_ns = -1
        # self.evt_nb = -1
        # self.run_nb = -1
        # self.nb_du = 0
        # self.du_pos = np.empty((self.du_count, 3), dtype=np.float32)
        self.f_name = ""

    def load_event_idx(self, idx):
        """
        Load event/run with index idx in list return by get_list_of_events()

        :param idx:
        """
        if self.idx_event == idx:
            return
        self.idx_event = idx
        self.event_number = self.l_events[idx][0]
        self.run_number = self.l_events[idx][1]
        self._load_event_identifier(self.event_number, self.run_number)

    def load_next_event(self):
        """
        Load next event, return False at the end of list else True
        """
        idx = self.idx_event + 1
        if idx >= self.get_nb_events():
            logger.warning(f"No more event")
            return False
        return self.load_event_idx(idx)

    def _load_event_identifier(self, event_number, run_number):
        """
        Load traces/pos of event/run event_number/run_number
        :param event_number:
        :param run_number:
        """
        logger.info(f"load event: {event_number} of run  {run_number}")
        # efield
        self.tt_event.get_event(event_number, run_number)
        self.du_id = self.tt_event.du_id
        self.du_count = self.tt_event.du_count

        # trace_size = len(self.tt_event.trace_x[0])
        trace_size = np.asarray(self.tt_event.trace).shape[-1]

        # if trace_size != self.traces.shape[2]:
        #    self.traces = np.empty((self.du_count, 3, trace_size), dtype=np.float32)
        #    logger.info(f"resize numpy array trace to {self.traces.shape}")
        self.traces = np.asarray(self.tt_event.trace)
        self.sig_size = self.traces.shape[-1]
        # self.traces[:, 0, :] = np.array(self.tt_event.trace_x, dtype=np.float32)
        # self.traces[:, 1, :] = np.array(self.tt_event.trace_y, dtype=np.float32)
        # self.traces[:, 2, :] = np.array(self.tt_event.trace_z, dtype=np.float32)
        # self.du_pos = np.empty((self.du_count, 3), dtype=np.float32)
        # self.du_pos[:, 0] = np.array(self.tt_event.pos_x, dtype=np.float32)
        # self.du_pos[:, 1] = np.array(self.tt_event.pos_y, dtype=np.float32)
        # self.du_pos[:, 2] = np.array(self.tt_event.pos_z, dtype=np.float32)

    def get_du_count(self):
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
        return 1e3 / self.t_bin_size
    
    def get_du_nanosec_ordered(self):
        """
        return nanosecond between 0s to 2s max
        """
        du_s = np.array(self.tt_event.du_seconds)
        min_sec = du_s.min()
        du_ns =  np.array(self.tt_event.du_nanoseconds) + 1000000000*(du_s-min_sec)
        return du_ns

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent Handling3dTracesOfEvent
        """
        o_tevent = Handling3dTracesOfEvent(
            f"{self.f_name} evt={self.event_number} run={self.run_number}"
        )
        # TODO: difference between du_id for all network and for specific event ?
        du_id = np.array(self.tt_event.du_id)
        o_tevent.init_traces(
            self.traces,
            du_id,
            self.get_du_nanosec_ordered(),
            self.get_sampling_freq_mhz(),
        )
        return o_tevent


class FileEfield(_FileEventBase):
    """
    File simulation of air shower with 4 TTree

      * tefield
      * trun
      * trunefieldsim
      * tshower

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

    """

    def __init__(self, f_name, check=True):
        """
        Constructor
        """
        name_ttree = "tefield"
        if check:
            if not os.path.exists(f_name):
                logger.error(f"File {f_name} doesn't exist.")
                raise FileNotFoundError
            if not check_ttree_in_file(f_name, name_ttree):
                logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
                raise AssertionError
        logger.info(f"Events  in file {f_name}")
        event = groot.TEfield(f_name)
        super().__init__(event)
        self.tt_efield = self.tt_event
        self.tt_shower = groot.TShower(f_name.replace('TEfield_', 'TShower_'))
        self.tt_run = groot.TRun(f_name.replace('TEfield_', 'TRun_'))
        self.load_event_idx(0)
        self.f_name = f_name

    def _load_event_identifier(self, event_number, run_number):
        """
        synchronize all event TTree with method get_event() in file.

        :param event_number:
        :param run_number:
        """
        super()._load_event_identifier(event_number, run_number)
        # synchronize showerTree on same evt, run
        logger.info(f"load tt_shower: {event_number} of run  {run_number}")
        self.tt_shower.get_event(event_number, run_number)
        # synchronize runTree on same run
        self.tt_run.get_run(run_number)
        # synchronize EfieldRunSimdataTree on same run
        self.t_bin_size = np.asarray(self.tt_run.t_bin_size)
        self.du_xyz = np.asarray(self.tt_run.du_xyz)
        

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.init_network(self.du_xyz)
        o_tevent.set_unit_axis(r"$\mu$V/m", "cart")
        return o_tevent


class FileVoltage(_FileEventBase):
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
        name_ttree = "tvoltage"
        if check:
            if not os.path.exists(f_name):
                logger.error(f"File {f_name} doesn't exist.")
                raise FileNotFoundError
            if not check_ttree_in_file(f_name, name_ttree):
                logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
                raise AssertionError
        logger.info(f"Events  in file {f_name}")
        event = groot.TVoltage(f_name)
        super().__init__(event)
        self.load_event_idx(0)
        self.f_name = f_name

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis(r"$\mu$V", "dir")
        return o_tevent


def get_file_event(f_name):
    """
    Factory for ROOT event file, return an instance of FileSimuEfield or FileVoltageEvent
    """
    if not os.path.exists(f_name):
        print("File does not exist")
        logger.error(f"File {f_name} doesn't exist.")
        raise FileNotFoundError
    trees_list = get_ttree_in_file(f_name)
    if "tefield" in trees_list:  # File with Efield info as input
        return FileEfield(f_name, False)
    if "tvoltage" in trees_list:  # File with voltage info as input
        return FileVoltage(f_name, False)
    logger.error(
        f"File {f_name} doesn't content TTree teventefield or teventvoltage. It contains {trees_list}."
    )
    raise AssertionError
