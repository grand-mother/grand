"""
Manage ROOT file of type Efield simulation, voltage ADCevent is experimental 
"""

import os.path
from logging import getLogger
from typing import Optional

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

class FileEvent:
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
        self.run_number: Optional[None, int]         = None
        self.event_number: Optional[None, int]       = None
        self.traces: Optional[None, np.ndarray]      = None
        self.sig_size: Optional[None, int]           = None
        self.t_bin_size: Optional[float]             = 0.5
        self.du_id: Optional[None, np,ndarray, list] = None
        self.du_count: Optional[None, int]           = None
        self.du_xyz: Optional[None, np.ndarray]      = None
        self.f_name: Optional[str] = ""
        self.tag: Optional[str]    = ""

        self.tt_event = tt_event
        self.l_events = self.tt_event.get_list_of_events()
        self.traces = np.empty((0, 3, 0), dtype=np.float32)
        self.idx_event = -1
        #self.delta_t_ns = -1
        #self.evt_nb = -1
        #self.run_nb = -1
        #self.nb_du = 0
        #self.du_pos = np.empty((self.du_count, 3), dtype=np.float32)
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
        self.du_id    = self.tt_event.du_id
        self.du_count = self.tt_event.du_count
        
        #trace_size = len(self.tt_event.trace_x[0])
        trace_size = np.asarray(self.tt_event.trace).shape[-1]

        #if trace_size != self.traces.shape[2]:
        #    self.traces = np.empty((self.du_count, 3, trace_size), dtype=np.float32)
        #    logger.info(f"resize numpy array trace to {self.traces.shape}")
        self.traces = np.asarray(self.tt_event.trace)
        self.sig_size = self.traces.shape[-1]
        #self.traces[:, 0, :] = np.array(self.tt_event.trace_x, dtype=np.float32)
        #self.traces[:, 1, :] = np.array(self.tt_event.trace_y, dtype=np.float32)
        #self.traces[:, 2, :] = np.array(self.tt_event.trace_z, dtype=np.float32)
        #self.du_pos = np.empty((self.du_count, 3), dtype=np.float32)
        #self.du_pos[:, 0] = np.array(self.tt_event.pos_x, dtype=np.float32)
        #self.du_pos[:, 1] = np.array(self.tt_event.pos_y, dtype=np.float32)
        #self.du_pos[:, 2] = np.array(self.tt_event.pos_z, dtype=np.float32)

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

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent Handling3dTracesOfEvent
        """
        o_tevent = Handling3dTracesOfEvent(f"{self.f_name} evt={self.event_number} run={self.run_number}")
        # TODO: difference between du_id for all network and for specific event ?
        du_id = np.array(self.tt_event.du_id)
        o_tevent.init_traces(
            self.traces,
            du_id,
            np.array(self.tt_event.du_nanoseconds),
            self.get_sampling_freq_mhz(),
        )
        o_tevent.init_network(self.du_xyz, du_id)
        return o_tevent

class File:
    """
    Simulation file with Efield or Voltage information.
    """
    """
    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributes:

        * event object: object ROOT_tree of type TTree Event
        * f_name str: path/name to ROOT file
        * trees_list list : list of tuple (event, run)
        * traces float(du_count,3,sig_size): 3D trace of N detector unit with S samples
        * traces_time float(du_count,sig_size): time bins of traces for each antenna.
        * sig_size int: number of time bins in traces
        * t_bin_size float: [ns] time sampling
        * event_number int: given by l_events list(idx_event)
        * run_number int: given by l_events list(idx_event)
        * du_id array: ID of DU in current event
        * du_count int: number of DU in current object. len(du_id).
        * du_xyz float(N,3): cartesian position of detector unit
    """

    def __init__(self, f_name):
        """
        Constructor for Efield and voltage event.
        """
        self.run_number: Optional[None, int]         = None
        self.event_number: Optional[None, int]       = None
        self.traces: Optional[None, np.ndarray]      = None
        self.traces_time: Optional[None, np.ndarray] = None
        self.sig_size: Optional[None, int]           = None
        self.t_bin_size: Optional[float]             = 0.5
        self.du_id: Optional[None, np,ndarray, list] = None
        self.du_count: Optional[None, int]           = None
        self.du_xyz: Optional[None, np.ndarray]      = None
        self.f_name: Optional[str] = ""
        self.tag: Optional[str]    = ""

        if not os.path.exists(f_name):
            logger.error(f"File {f_name} doesn't exist.")
            raise FileNotFoundError
        else:
            self.f_name = f_name
            tfile   = ROOT.TFile.Open(f_name)
            l_ttree = tfile.GetListOfKeys()
            self.trees_list = [ttree.GetName() for ttree in l_ttree]
            if "tefield" in self.trees_list:          # File with Efield info as input
                self.tag     = "efield"
                self.event   = groot.TEfield(f_name)
                self.shower  = groot.TShower(f_name)
                self.run     = groot.TRun(f_name)
            elif "tvoltage" in self.trees_list:       # File with voltage info as input
                self.tag   = "voltage"
                self.event = groot.TVoltage(f_name)
            else:
                logger.error(f"File {f_name} doesn't contain TTree tefield or tvoltage. It contains {self.trees_list}.")
                raise AssertionError

            self.event_list = self.event.get_list_of_events() # [[evt0, run0], [evt1, run0], ...[evt0, runN], ...]
            self.get_event(event_idx=0)  # initialize instance to the first event

        logger.info(f"Events  in file {f_name}")

    def get_event(self, event_idx=0):
        """
        Load traces/pos of event/run event_number/run_number
        :param event_number:
        :param run_number:
        """
        self.event_idx: int = event_idx  # index of events in the input file. 0 for first event and so on.
        if self.event_idx<len(self.event_list):
            self.event_number = self.event_list[self.event_idx][0] 
            self.run_number = self.event_list[self.event_idx][1]
        else:
            logger.warning(f"Event index {self.event_idx} is out of range. It must be less than {len(self.event_list)}.")
            return False

        logger.info(f"load event: {self.event_number} of run  {self.run_number}")
        self.event.get_event(self.event_number, self.run_number)
        self.traces = np.asarray(self.event.trace)
        self.sig_size = self.traces.shape[-1]
        self.du_id  = np.asarray(self.event.du_id)
        self.du_count  = self.event.du_count
        if self.tag=="efield":
            self.shower.get_event(self.event_number, self.run_number)
            self.run.get_run(self.run_number)
            self.du_xyz = np.asarray(self.run.du_xyz)
            self.t_bin_size = self.run.t_bin_size

        self.traces_time = self.get_traces_time()

    def get_next_event(self):
        """
        Load next event, return False at the end of list else True
        """
        self.event_idx += 1
        if self.event_idx<len(self.event_list): 
            return self.get_event(self.event_idx)
        else:
            logger.warning(f"Event index {self.event_idx} is out of range. It must be less than {len(self.event_list)}.")
            return False

    def get_traces_time(self):
        """
        Define time sample in ns for the duration of the trace
        t_samples.shape  = (du_count, self.sig_size)
        t_start_ns.shape = (du_count,)
        """
        t_start_ns = np.asarray(self.event.du_nanoseconds)[...,np.newaxis]  # shape = (du_count, 1)

        outer_product = np.outer(self.t_bin_size * np.ones(self.du_count), np.arange(0, self.sig_size, dtype=np.float64))
        t_samples = outer_product + t_start_ns
        logger.info(f"shape du_nanoseconds and t_samples =  {t_start_ns.shape}, {t_samples.shape}")

        return t_samples

class FileSimuEfield(FileEvent):
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

      * same as FileEvent class
      * tt_efield object EfieldEventTree
      * tt_shower object ShowerEventSimdataTree
      * tt_run object RunTree

    """

    def __init__(self, f_name):
        """
        Constructor
        """
        name_ttree = "tefield"
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
        self.tt_shower = groot.TShower(f_name)
        self.tt_run    = groot.TRun(f_name)
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
        self.t_bin_size = self.tt_run.t_bin_size
        self.du_xyz = np.asarray(self.tt_run.du_xyz)

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis("$\mu$V/m", "cart")
        return o_tevent


class FileVoltageEvent(FileEvent):
    """
    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributs:

      * same as FileEvent class
    """

    def __init__(self, f_name):
        """
        Constructor
        """
        name_ttree = "tvoltage"
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
        o_tevent.set_unit_axis("$\mu$V", "dir")
        return o_tevent


class FileADCeventProto(FileEvent):
    """

    .. warning: Prototype for ADC event, work in progress


    Goals of the class:

      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    Public attributes:

      * same as FileEvent class
      * all_traces array(nb_all_event_in_file, nb_sample)
    """

    def __init__(self, f_name):
        """
        Constructor
        """
        self.warning = True
        name_ttree = "tadc"
        if not os.path.exists(f_name):
            logger.error(f"File {f_name} doesn't exist.")
            raise FileNotFoundError
        if not check_ttree_in_file(f_name, name_ttree):
            logger.error(f"File {f_name} doesn't content TTree {name_ttree}")
            raise AssertionError
        logger.info(f"Events  in file {f_name}")
        #event = groot.ADCEventTree(f_name)
        event = groot.TADC(f_name)
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

    def _load_event_identifier(self, event_number, run_number):
        """
        Load traces/pos of event/run event_number/run_number

        :param event_number:
        :param run_number:
        """
        logger.info(f"load event: {event_number} of run  {run_number}")
        # efield
        self.tt_event.get_event(event_number, run_number)
        self.du_count = self.tt_event.du_count
        trace_size = len(self.tt_event.trace_0[0])
        if trace_size != self.traces.shape[2]:
            self.traces = np.zeros((self.du_count, 3, trace_size), dtype=np.float32)
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
        self.du_xyz = np.empty((self.du_count, 3), dtype=np.float32)
        self.du_xyz[:, 0] = np.array(self.tt_event.gps_long, dtype=np.float32)
        self.du_xyz[:, 1] = np.array(self.tt_event.gps_lat, dtype=np.float32)
        self.du_xyz[:, 2] = np.array(self.tt_event.gps_alt, dtype=np.float32)



