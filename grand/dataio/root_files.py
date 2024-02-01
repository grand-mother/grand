"""
Manage ROOT file of type event : Efield, voltage
The main functionnalities of this module is 
* to synchronize the relevant TTree (tshower , trun) on the same event
* provide a numpy container of traces each a specific event, with method get_obj_handling3dtraces()

"""

import os.path
from logging import getLogger
from typing import Optional
import glob

import numpy as np
import ROOT

import grand.dataio.root_trees as groot
from grand.basis.traces_event import Handling3dTraces

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


def get_file_event(f_name):
    """
    Factory for ROOT event file, Efield or voltage
    """
    if not os.path.exists(f_name):
        logger.error(f"File {f_name} doesn't exist.")
        raise FileNotFoundError
    trees_list = get_ttree_in_file(f_name)
    if "tefield" in trees_list:  # File with Efield info as input
        return FileEfield(f_name)
    if "tvoltage" in trees_list:  # File with voltage info as input
        return FileVoltage(f_name)
    logger.error(
        f"File {f_name} doesn't content TTree teventefield or teventvoltage. It contains {trees_list}."
    )
    raise AssertionError


def _get_root_file(f_name, prefix):
    dir_root = os.path.dirname(f_name)
    l_froot = glob.glob(dir_root + "/*root")
    if len(l_froot) == 0:
        logger.error("No ROOT file")
        raise AssertionError
    for n_file in l_froot:
        ns_file = n_file.split("/")[-1]
        if ns_file.find(prefix) == 0:
            logger.info(f"{prefix} file is {n_file}")
            return n_file
    logger.error(f"No ROOT {prefix} file")
    raise AssertionError


def get_trun(f_name):
    return _get_root_file(f_name, "trun_")


def get_tshower(f_name):
    return _get_root_file(f_name, "tshower_")


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

    def __init__(self, tt_event, f_name):
        """
        Constructor
        """
        self.run_number: Optional[None, int] = None
        self.event_number: Optional[None, int] = None
        self.traces: Optional[None, np.ndarray] = None
        self.sig_size: Optional[None, int] = None
        self.t_bin_size: Optional[float] = 0.5
        self.du_id: Optional[None, np.ndarray, list] = None
        self.du_count: Optional[None, int] = None
        self.du_xyz: Optional[None, np.ndarray] = None
        self.f_name: Optional[str] = ""
        self.tag: Optional[str] = ""
        self.tt_event = tt_event
        self.l_events = self.tt_event.get_list_of_events()
        self.traces = np.empty((0, 3, 0), dtype=np.float32)
        self.idx_event = -1
        self.tt_shower = groot.TShower(get_tshower(f_name))
        self.tt_run = groot.TRun(get_trun(f_name))
        self.f_name = f_name
        self.load_event_idx(0)
    
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
        self.tt_event.get_event(event_number, run_number)
        self.du_id = self.tt_event.du_id
        self.du_count = self.tt_event.du_count
        self.tt_shower.get_event(event_number, run_number)
        self.tt_run.get_run(run_number)
        self.t_bin_size = np.asarray(self.tt_run.t_bin_size)
        self.du_xyz = np.asarray(self.tt_run.du_xyz)
        self.traces = np.asarray(self.tt_event.trace)
        self.sig_size = self.traces.shape[-1]

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
        du_ns = np.array(self.tt_event.du_nanoseconds) + 1000000000 * (du_s - min_sec)
        return du_ns

    def get_obj_handling3dtraces(self):
        """
        Return a traces container IO independent Handling3dTraces
        """
        o_tevent = Handling3dTraces(
            f"{self.f_name}\nevt_nb={self.event_number} run_nb={self.run_number}"
        )
        du_id = np.array(self.tt_event.du_id)
        o_tevent.init_traces(
            self.traces,
            du_id,
            self.get_du_nanosec_ordered(),
            self.get_sampling_freq_mhz(),
        )
        o_tevent.init_network(self.du_xyz)
        o_tevent.network.name = self.tt_run.site
        shw = self.tt_shower
        xmax = shw.xmax_pos_shc
        dist_xmax = np.linalg.norm(xmax) / 1000
        o_tevent.info_shower = f"||xmax_pos_shc||={dist_xmax:.1f} km;"
        azi, zenith = shw.azimuth, shw.zenith
        o_tevent.info_shower += f" (azi, zenith)=({azi:.0f}, {zenith:.0f}) deg;"
        nrj = shw.energy_primary
        o_tevent.info_shower += f" energy_primary={nrj:.1e} GeV"
        return o_tevent


class FileEfield(_FileEventBase):
    """
    Goals of the class:

      * Event type is Efield
    """

    def __init__(self, f_name):
        """
        :param f_name: path to ROOT file Efield
        """
        event = groot.TEfield(f_name)
        super().__init__(event, f_name)
        self.tt_efield = self.tt_event

    def get_obj_handling3dtraces(self):
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis(r"$\mu$V/m", "cart")
        o_tevent.type_trace = "Efield"
        return o_tevent


class FileVoltage(_FileEventBase):
    """
    Goals of the class:

      * Event type is voltage
    """

    def __init__(self, f_name):
        """

        :param f_name:  path to ROOT file volatge
        """
        event = groot.TVoltage(f_name)
        super().__init__(event, f_name)
        self.tt_volt = self.tt_event

    def get_obj_handling3dtraces(self):
        o_tevent = super().get_obj_handling3dtraces()
        o_tevent.set_unit_axis(r"$\mu$V", "dir")
        o_tevent.type_trace = "Voltage"
        return o_tevent
