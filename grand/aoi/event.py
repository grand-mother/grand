# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field, fields

import numpy as np
import ROOT
from pathlib import Path
import shutil

from grand import CartesianRepresentation
from grand.aoi.timetrace import Voltage, Efield, TreeExists
from grand.aoi.antenna import Antenna
from grand.aoi.shower import Shower
from grand.dataio import DataDirectory, TRun, TRunRawVoltage, TVoltage, TEfield, TShower, TRawVoltage, grand_tree_list, NotUniqueEvent
import grand.dataio
from line_profiler import profile


@dataclass
class Event:
    """A class for holding an event"""

    # ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    _file: ROOT.TFile = None
    """The instance of the file with TTrees containing the event."""

    _directory: DataDirectory = None
    """The instance of the directory with files with TTrees containing the event."""

    event_number: int = None
    """The current event in the file number"""

    run_number: int = None
    """The run number of the current event"""

    _entry_number: int = None
    """The entry number - used for enforcing loading specific entry from all the event trees. Makes sense only if those trees have all the same events."""

    # antennas: list[Antenna] = None
    antennas: list = None
    """Antennas participating in the event"""

    _all_antennas: dict = None
    """All antennas in the run - a workaround for lack of antennas in GP300"""

    # voltages: list[Voltage] = None
    voltages: list = None
    """Voltages from different antennas"""

    # efields: list[Efield] = None
    efields: list = None
    """Efields from different antennas"""

    shower: Shower() = None
    """Reconstructed shower"""

    simshower: Shower() = None
    """Simulated shower for simulations"""

    ## ToDo: what is it?
    L: int = 0
    """Event multiplicity"""

    ## Time vector - from the start of singla in first DU to the end in last DU
    t_vector: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    """Time vector - from the start of singla in first DU to the end in last DU"""

    # Reconstruction parameters

    is_reconstructed: bool = False
    """Was this event reconstructed"""

    is_wave: bool = False
    """Is this event associated to a single wave based on reconstruction"""

    origin_planewave: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    """Vector of origin of plane wave fit"""

    chi2_planewave: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    """Chi2 of plane wave fit"""

    origin_sphere: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    """Position of the source according to spherical fit"""

    chi2_sphere: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    """Chi2 of spherical fit"""

    is_eas: bool = False
    """Is this an EAS?"""

    # *** Run related properties
    ## ToDo: should get enum description for that, but I don't think it exists at the moment
    run_mode: np.uint32 = 0
    """Run mode - calibration/test/physics."""

    # ToDo: list of readable events should be held somewhere in this interface, but where?
    ## Run's first event
    # _first_event: np.ndarray = np.zeros(1, np.uint32)
    ## First event time
    # _first_event_time: np.ndarray = np.zeros(1, np.uint32)
    ## Run's last event
    # _last_event: np.ndarray = np.zeros(1, np.uint32)
    ## Last event time
    # _last_event_time: np.ndarray = np.zeros(1, np.uint32)
    # These are not from the hardware

    data_source: str = "other"
    """Data source, detector, sim, other"""

    data_generator: str = "GRANDlib"
    """Data generator, gtot (in this case)"""

    data_generator_version: str = "0.1.0"
    """Generator version, gtot version (in this case)"""

    site: str = "DummySite"
    """Site name"""

    # ## Site longitude
    # site_long: np.float32 = 0
    # ## Site latitude
    # site_lat: np.float32 = 0

    _origin_geoid: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    """Origin of the coordinate system used for the array"""

    _t_bin_size: int = 2
    """Time bin size [ns]"""

    # Internal trees
    trun: TRun = None
    """DOI's TRun tree containing all run information"""

    trunrawvoltage: TRunRawVoltage = None
    """DOI's TRunRawVoltage tree containing voltage run information"""

    tvoltage: TVoltage = None
    """DOI's TVoltage/TRawVoltage tree containing voltage information"""

    tefield: TEfield = None
    """DOI's TEfield tree containing Efield information"""

    tshower: TShower = None
    """DOI's TShower tree containing reconstructed shower information"""

    tsimshower: TShower = None
    """DOI's TShower tree containing simulated shower information"""

    # Tree files

    file_trun: ROOT.TFile = None
    """TRun file"""

    file_trunrawvoltage: ROOT.TFile = None
    """TRunRawVoltage file"""

    file_tvoltage: ROOT.TFile = None
    """TVoltage file"""

    file_tefield: ROOT.TFile = None
    """TEfield file"""

    file_tshower: ROOT.TFile = None
    """TShower file"""

    file_tsimshower: ROOT.TFile = None
    """TSimShower file"""

    is_starshape: bool = False
    """Is this event a star shape?"""

    # Options

    auto_file_close: bool = True
    """Close files automatically after event write? - slower writing but less maitanance by the user"""

    # Lists of trees
    _run_trees: list = None
    _event_trees: list = None
    _trees: list = None

    # Choose the level of the efield
    tefield_level: int  = None

    ## Post-init actions, like an automatic readout from files, etc.
    def __post_init__(self):
        # If the file name was given, init the Event from trees
        if self._file:
            self.fill_event_from_trees()
            self.fill_t_vector()

    @property
    def file(self):
        """A single file that contains all the TTrees"""
        return str(self._file)

    @file.setter
    def file(self, value):
        """A single file that contains all the TTrees"""

        # If the _file is not yet TFile, make it so
        if not isinstance(value, ROOT.TFile):
            self._file = ROOT.TFile(value, "read")
        else:
            self._file = value

        # Set all the tree files as this file
        self.file_trun = self._file
        self.file_trunrawvoltage = self._file
        self.file_tvoltage = self._file
        self.file_tefield = self._file
        self.file_tshower = self._file
        self.file_tsimshower = self._file

    @property
    def directory(self):
        """A single file that contains all the TTrees"""
        return self._directory

    @directory.setter
    def directory(self, value):
        """A directory that contains all the files with TTrees"""
        # If the _file is not yet TFile, make it so
        if not isinstance(value, DataDirectory):
            self._directory = DataDirectory(value)
        else:
            self._directory = value

        # Set all the tree files as this file and trees as file's trees
        self.file_trun = self.directory.ftrun.f
        self.trun = self.directory.trun
        if self.directory.ftrunrawvoltage:
            self.file_trunrawvoltage = self.directory.ftrunrawvoltage.f
            self.trunrawvoltage = self.directory.trunrawvoltage
        if self.directory.ftvoltages:
            self.file_tvoltage = self.directory.ftvoltage.f
            self.tvoltage = self.directory.tvoltage
        if self.directory.ftefield:
            self.file_tefield = self.directory.ftefield.f
            # If the efield level was not specified, use the default one
            if self.tefield_level is None:
                self.tefield = self.directory.tefield
            else:
                self.tefield = getattr(self.directory, f"tefield_l{self.tefield_level}")
        if self.directory.ftshower_l1:
            self.file_tshower = self.directory.ftshower_l1.f
            self.tshower = self.directory.tshower_l1
        if self.directory.ftshower_l0:
            self.file_tsimshower = self.directory.ftshower_l0.f
            self.tsimshower = self.directory.tshower_l0
        if self.directory.ftrawvoltages and not self.file_tvoltage:
            self.file_tvoltage = self.directory.ftrawvoltage.f
            self.tvoltage = self.directory.trawvoltage



    @property
    def origin_geoid(self):
        """Origin of the coordinate system used for the array"""
        return self._origin_geoid

    @origin_geoid.setter
    def origin_geoid(self, v):
        self._origin_geoid = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    ## Fill this event from trees
    def fill_event_from_trees(self, event_number=None, run_number=None, entry_number=None, simshower=False, use_trawvoltage=False, trawvoltage_channels=[0,1,2], init_trees=True, gp300_workaround=True, tefield_level=None):
        """Fill this event from trees
        :param simshower: how to treat the TShower existing in the file, as sim values or reconstructed values
        :type simshower: bool
        """
        # Check if any of the files exist
        if not self._file and not self.file_trun and not self.file_trunrawvoltage and not self.file_tvoltage and not self.file_tefield and not self.file_tshower and not self.file_tsimshower:
            print("No files provided to init from. Aborting.")
            return False

        # *** Set the run/event/entry number if requested.

        # If entry/event/run number not specified, take the first entry
        run_entry_number = None
        if self._entry_number is None and self.run_number is None and self.event_number is None and entry_number is None and run_number is None and event_number is None:
            entry_number = 0
            run_entry_number = 0

        # Don't allow specifying entry and event/run at the same time, because... what to chose?
        if entry_number is not None and (run_number is not None or event_number is not None):
            print("Please provide only entry_number or event/run_number!")
        if entry_number is not None:
            self._entry_number = entry_number
            # ToDo: this should be run number from an even tree with entry_number...
            if run_entry_number is None and self.run_number is None:
                run_entry_number = 0
        else:
            if run_number is not None:
                self.run_number = run_number
            if event_number is not None:
                self.event_number = event_number
                self._entry_number = None

        # *** Check what TTrees are available and fill according to their availability

        # If initialising trees requested
        if init_trees:
            # Check the Run tree existence
            if trun := self.file_trun.Get("trun"):
                self.trun = TRun(_tree=trun)
            else:
                print("No Run tree. Run information will not be available.")
                # Make trun really None
                self.trun = None

        # If self.trun was successfully initialised
        if self.trun is not None:
            # Fill part of the event from trun
            ret = self.fill_event_from_runtree(run_entry_number=run_entry_number)
            if ret:
                print("Run information loaded.")
            else:
                print("No Run tree. Run information will not be available.")

        # Check the TRunRawVoltage file existence
        if self.file_trunrawvoltage is not None:
            # If initialising trees requested
            if init_trees:
                # Check the TRunRawVoltage tree existence
                if trunrawvoltage := self.file_trunrawvoltage.Get("trunrawvoltage"):
                    self.trunrawvoltage = TRunRawVoltage(_tree=trunrawvoltage)
                else:
                    print("No TRunRawVoltage tree. RunRawVoltage information will not be available.")
                    # Make trunrawvoltage really None
                    self.trunrawvoltage = None

        # If self.trunrawvoltage was successfully initialised
        if self.trunrawvoltage is not None:
            # Fill part of the event from trunrawvoltage
            ret = self.fill_event_from_runrawvoltagetree(run_entry_number=run_entry_number)
            if ret:
                print("RunRawVoltage information loaded.")
            else:
                print("No RunRawVoltage tree. RunRawVoltage information will not be available.")

        if self.file_tvoltage:
            # Use standard voltage tree
            if not use_trawvoltage:
                # If initialising trees requested
                if init_trees:
                    # Check the Voltage tree existence
                    if tvoltage := self.file_tvoltage.Get("tvoltage"):
                        self.tvoltage = TVoltage(_tree=tvoltage)
                    else:
                        # print("No Voltage tree. Voltage information will not be available.")
                        # Make tvoltage really None
                        self.tvoltage = None

                # If self.tvoltage was successfully initialised
                if self.tvoltage is not None:
                    # Fill part of the event from tvoltage
                    ret = self.fill_event_from_voltage_tree()
                    if ret:
                        print("Voltage information loaded.")
                    else:
                        # print("No Voltage tree. Voltage information will not be available.")
                        # Make tvoltage really None
                        self.tvoltage = None

            # Use trawvoltage tree if requested or tvoltage tree not found
            if use_trawvoltage or self.tvoltage==None:
                # If initialising trees requested
                if init_trees:
                    # Check the Voltage tree existence
                    if tvoltage := self.file_tvoltage.Get("trawvoltage"):
                        self.tvoltage = TRawVoltage(_tree=tvoltage)
                        use_trawvoltage = True
                    else:
                        print("No Voltage or TRawVoltage tree. Voltage information will not be available.")
                        # Make tvoltage really None
                        self.tvoltage = None

                # If self.tvoltage was successfully initialised
                if self.tvoltage is not None:
                    # Fill part of the event from tvoltage
                    ret = self.fill_event_from_voltage_tree(use_trawvoltage=use_trawvoltage, trawvoltage_channels=trawvoltage_channels)
                    if ret:
                        print("Voltage information (from TRawVoltage) loaded.")
                    else:
                        print("No Voltage or TRawVoltage tree. Voltage information will not be available.")
                        # Make tvoltage really None
                        self.tvoltage = None

        # Check the Efield file existence
        if self.file_tefield:
            # If initialising trees requested
            if init_trees:
                tree_name = "tefield"
                # If specific tree level was requested
                if tefield_level:
                    tree_name += f"_{tefield_level}"
                # Check the Efield tree existence
                if tefield := self.file_tefield.Get(tree_name):
                    self.tefield = TEfield(_tree=tefield)
                else:
                    print("No Efield tree. Efield information will not be available.")
                    # Make tefield really None
                    self.tefield = None

            # If self.tefield was successfully initialised
            if self.tefield is not None:
                # Fill part of the event from tefield
                ret = self.fill_event_from_efield_tree()
                if ret:
                    print("Efield information loaded.")
                else:
                    print("No Efield tree. Efield information will not be available.")
                    # Make tefield really None
                    self.tefield = None

        # Check the Shower file existence
        if self.file_tshower or self.file_tsimshower:
            # If initialising trees requested
            if init_trees:
                # Check the Shower tree existence
                if tshower := self.file_tshower.Get("tshower"):
                    if simshower:
                        self.tsimshower = TShower(_tree=tshower)
                    else:
                        self.tshower = TShower(_tree=tshower)
                else:
                    print("No Shower tree. Shower information will not be available.")
                    # Make tshower really None
                    if simshower:
                        self.tsimshower = None
                    else:
                        self.tshower = None

            # If self.t(sim)shower was successfully initialised
            if (simshower and self.tsimshower is not None) or (not simshower and self.tshower is not None):
                # Fill part of the event from tshower
                ret = self.fill_event_from_shower_tree(simshower)
                if ret:
                    print("Shower information loaded.")
                else:
                    print("No Shower tree. Shower information will not be available.")
                    # Make tshower really None
                    if simshower:
                        self.tsimshower = None
                    else:
                        self.tshower = None

        # Check the sim Shower file existence
        if self.file_tsimshower:
            # If initialising trees requested
            if init_trees:
                # Check the SimShower tree existence
                if tsimshower := self.file_tsimshower.Get("tshower"):
                    self.tsimshower = TShower(_tree=tsimshower)
                else:
                    print("No Simulated Shower tree. Simulated Shower information will not be available.")
                    # Make tsimshower really None
                    self.tsimshower = None

            # If self.tsimshower was successfully initialised
            if self.tsimshower is not None:
                # Fill part of the event from tshower
                ret = self.fill_event_from_shower_tree(True)
                if ret:
                    print("Simulated shower information loaded.")
                else:
                    print("No Simulated shower tree. Simulated shower information will not be available.")
                    # Make tsimshower really None
                    self.tsimshower = None

        self.fill_antennas(gp300_workaround=True)

        # Set the event number and run number in somewhat ugly way - from the first non None tree
        for t in [self.tvoltage, self.tefield, self.tshower, self.tsimshower]:
            if t is not None:
                self.event_number = t.event_number
                self.run_number = t.run_number
                break

        # Fill the time vector
        self.fill_t_vector()

        # Fill the tree lists
        self._run_trees = [self.trun, self.trunrawvoltage]
        self._event_trees = [self.tvoltage, self.tefield, self.tshower, self.tsimshower]
        self._trees = self._run_trees + self._event_trees

    ## Fill part of the event from the Run tree
    def fill_event_from_runtree(self, run_entry_number=None):
        ret = 1

        # For star shape, the run entry number should be the same as event entry number
        if self.is_starshape and run_entry_number is None and self.run_number is None:
            run_entry_number = self._entry_number

        # If run number not provided in any way, get the first entry
        if run_entry_number is None and self.run_number is None:
            run_entry_number = 0

        # Read the event into the class
        if run_entry_number is None:
            ret = self.trun.get_run(self.run_number)
        else:
            ret = self.trun.get_entry(run_entry_number)

        # Copy the values
        self.run_mode = self.trun.run_mode
        self.data_source = self.trun.data_source
        self.data_generator = self.trun.data_generator
        self.data_generator_version = self.trun.data_generator_version
        self.site = self.trun.site
        # self.site_long = self.trun.site_long
        # self.site_lat = self.trun.site_lat
        self.origin_geoid = self.trun.origin_geoid
        # ToDo: This assumes uniform t_bin_size (to avoid current mismatch in number of bins for different trees coming from sim2root)
        self._t_bin_size = self.trun.t_bin_size[0]

        # Check if the run is star shape
        if "star_shape" in self.trun.site_layout:
            self.is_starshape = True

        return ret

    ## Fill part of the event from the Run tree
    def fill_event_from_runrawvoltagetree(self, run_entry_number=None):
        # For star shape, the run entry number should be the same as event entry number
        if self.is_starshape and run_entry_number is None and self.run_number is None:
            run_entry_number = self._entry_number

        # If run number not provided in any way, get the first entry
        if run_entry_number is None and self.run_number is None:
            run_entry_number = 0

        # Read the event into the class
        if run_entry_number is None:
            ret = self.trunrawvoltage.get_run(self.run_number)
        else:
            ret = self.trunrawvoltage.get_entry(run_entry_number)

        return ret


    ## Fill event's antennas
    def fill_antennas(self, gp300_workaround=True):
        """Fill event's antennas"""
        self.antennas = []

        # For GP300 for now, get the GPS coordinates for each DU and calculate the x/y/z here
        if gp300_workaround and "GP300" in self.site or "GP80" in self.site or "GP13" in self.site:

            # Get the tree we are using
            cur_tree = None
            if self.tefield is not None:
                cur_tree = self.tefield
            elif self.tvoltage is not None:
                cur_tree = self.tvoltage
            else:
                raise "Can't calculate antennas positions"

            # If this is the first time we calculate antennas positions
            if not self._all_antennas:
                print("GP300 workaround: calculating all antennas positions")
                from grand import ECEF, Geodetic, GRANDCS

                # Get the coordinates for all DUs from all events
                count = cur_tree.draw("du_id:gps_lat:gps_long:gps_alt", "", "goff")
                if count == -1:
                    raise "Can't get antenna positions from the ROOT file"

                du_ids = np.array(np.frombuffer(cur_tree.get_v1(), dtype=np.float64, count=count)).astype(np.int32)
                du_lats = np.array(np.frombuffer(cur_tree.get_v2(), dtype=np.float64, count=count)).astype(np.float32)
                du_lons = np.array(np.frombuffer(cur_tree.get_v3(), dtype=np.float64, count=count)).astype(np.float32)
                du_alts = np.array(np.frombuffer(cur_tree.get_v4(), dtype=np.float64, count=count)).astype(np.float32)

                # Get indices of the unique du_ids
                # ToDo: sort?
                unique_dus_idx = np.unique(du_ids, return_index=True)[1]
                # Leave only the unique du_ids
                du_ids = du_ids[unique_dus_idx]
                du_lats = du_lats[unique_dus_idx]
                du_lons = du_lons[unique_dus_idx]
                du_alts = du_alts[unique_dus_idx]

                # Get lat/lon/alt from xyz
                origin = Geodetic(latitude=40.95068711, longitude=93.96977396, height=1200)

                geod_ant = Geodetic(latitude=du_lats, longitude=du_lons, height=du_alts)
                grandcs  = GRANDCS(geod_ant, location=origin)

                self._all_antennas = {}

                for i in range(len(du_ids)):
                    a = Antenna()
                    a.id = du_ids[i]
                    a.position.x = grandcs[0,i]
                    a.position.y = grandcs[1,i]
                    a.position.z = grandcs[2,i]
                    a.tilt.x = 0
                    a.tilt.y = 0

                    self._all_antennas[a.id] = a


            # Fill the antenna part
            event_dus = cur_tree.du_id
            if self._entry_number is not None:
                # ToDo: Handle ret
                ret = cur_tree.get_entry(self._entry_number)
            else:
                # ToDo: Handle ret
                ret = cur_tree.get_event(self.event_number, self.run_number)

            for du_id in event_dus:
                a = Antenna()
                a.id = du_id
                a.position.x = self._all_antennas[du_id].position.x
                a.position.y = self._all_antennas[du_id].position.y
                a.position.z = self._all_antennas[du_id].position.z
                a.tilt.x = self._all_antennas[du_id].tilt.x
                a.tilt.y = self._all_antennas[du_id].tilt.y

                self.antennas.append(a)


        else:
            # Fill the antenna part
            if self.tefield is not None: event_dus_indices = self.tefield.get_dus_indices_in_run(self.trun)
            elif self.tvoltage is not None: event_dus_indices = self.tvoltage.get_dus_indices_in_run(self.trun)
            for i in range(len(event_dus_indices)):
                a = Antenna()
                ant_ind = int(event_dus_indices[i])
                a.id = self.trun.du_id[ant_ind]
                a.position.x = self.trun.du_xyz[ant_ind][0]
                a.position.y = self.trun.du_xyz[ant_ind][1]
                a.position.z = self.trun.du_xyz[ant_ind][2]
                a.tilt.x = self.trun.du_tilt[ant_ind][0]
                a.tilt.y = self.trun.du_tilt[ant_ind][1]

                self.antennas.append(a)

                self._all_antennas = {}

            # ToDo: it seems that all antennas of the array may be needed in AOI, so perhaps they should be advanced from an internal variable
            for i in range(len(self.trun.du_id)):
                a = Antenna()
                a.id = self.trun.du_id[i]
                a.position.x = self.trun.du_xyz[i][0]
                a.position.y = self.trun.du_xyz[i][1]
                a.position.z = self.trun.du_xyz[i][2]
                a.tilt.x = 0
                a.tilt.y = 0

                self._all_antennas[a.id] = a



    ## Fill part of the event from the Voltage tree
    def fill_event_from_voltage_tree(self, use_trawvoltage=False, trawvoltage_channels=(0,1,2)):
        ret = 1
        if self._entry_number is not None:
            ret = self.tvoltage.get_entry(self._entry_number)
        else:
            ret = self.tvoltage.get_event(self.event_number, self.run_number)
        # self.tvoltage.get_entry(0)
        self.voltages = []

        # Get number of DUs
        if not use_trawvoltage:
            trace_cnt = len(self.tvoltage.trace)
        else:
            trace_cnt = len(self.tvoltage.trace_ch)

        # Obtain the start time of the earliest trace. ToDo: maybe the first trace in the file is always first in time? That would save time...
        min_t0 = np.min(np.array(np.array(self.tvoltage.du_seconds).astype(np.int64)*1000000000+np.array(self.tvoltage.du_nanoseconds).astype(np.int64), dtype="datetime64[ns]"))

        # Loop through traces
        for i in range(trace_cnt):
            # Fill the voltage trace part
            v = Voltage()
            # trr = self.tvoltage.trace[i]
            if not use_trawvoltage:
                trace = self.tvoltage.trace[i]
                # tx = self.tvoltage.trace[i][0]
                tx = trace[0]
            else:
                trace = self.tvoltage.trace_ch[i]
                # tx = self.tvoltage.trace_ch[i][trawvoltage_channels[0]]
                tx = trace[trawvoltage_channels[0]]
            v.n_points = len(tx)
            # ToDo: That's the trigger time for now, and should be the start time of the trace
            v.t0 = np.datetime64(self.tvoltage.du_seconds[i]*1000000000+self.tvoltage.du_nanoseconds[i], "ns")
            v.t_bin_size = self._t_bin_size
            # The default size of the CartesianRepresentation is wrong. ToDo: it should have some resize
            v.trace = CartesianRepresentation(x=np.zeros(len(tx), np.float64), y=np.zeros(len(tx), np.float64), z=np.zeros(len(tx), np.float64))
            v.trace.x = tx
            if not use_trawvoltage:
                v.trace.y = trace[1]
                v.trace.z = trace[2]
                # v.trace.y = self.tvoltage.trace[i][1]
                # v.trace.z = self.tvoltage.trace[i][2]
            else:
                # v.trace.y = self.tvoltage.trace_ch[i][trawvoltage_channels[1]]
                # v.trace.z = self.tvoltage.trace_ch[i][trawvoltage_channels[2]]
                v.trace.y = trace[trawvoltage_channels[1]]
                v.trace.z = trace[trawvoltage_channels[2]]

            # Generate the time array
            v.calculate_t_vector(min_t0)

            v.du_id = self.tvoltage.du_id[i]

            v.trigger_time = np.datetime64(self.tvoltage.du_seconds[i] * 1000000000 + self.tvoltage.du_nanoseconds[i], "ns")

            self.voltages.append(v)

        # ## The trace length
        # _n_points: int = 0
        # ## [ns] n_points x step = total timetrace length
        # _time_step: float = 0
        # ## Start time as unix time with nanoseconds
        # _t0: np.datetime64 = np.datetime64(0, 'ns')
        # ## Trigger time as unix time with nanoseconds
        # _trigger_time: np.datetime64 = np.datetime64(0, 'ns')
        #
        # ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
        # ## Hilbert envelope vector in X
        # _hilbert_trace_x: np.ndarray = np.zeros(1, np.float)
        # ## Hilbert envelope vector in X
        # _hilbert_trace_y: np.ndarray = np.zeros(1, np.float)
        # ## Hilbert envelope vector in X
        # _hilbert_trace_z: np.ndarray = np.zeros(1, np.float)

        return ret

    ## Fill part of the event from the Efield tree
    def fill_event_from_efield_tree(self):
        ret = 1
        if self._entry_number is not None:
            ret = self.tefield.get_entry(self._entry_number)
        else:
            ret = self.tefield.get_event(self.event_number, self.run_number)
        self.efields = []

        # Obtain the start time of the earliest trace. ToDo: maybe the first trace in the file is always first in time? That would save time...
        min_t0 = np.min(np.array(np.array(self.tefield.du_seconds).astype(np.int64) * 1000000000 + np.array(self.tefield.du_nanoseconds).astype(np.int64), dtype="datetime64[ns]"))

        # Loop through traces
        for i in range(len(self.tefield.trace)):
            v = Efield()
            trace = self.tefield.trace[i]
            # tx = self.tefield.trace[i][0]
            tx = trace[0]
            v.n_points = len(tx)
            v.t0 = np.datetime64(self.tefield.du_seconds[i] * 1000000000 + self.tefield.du_nanoseconds[i], "ns")
            # The default size of the CartesianRepresentation is wrong. ToDo: it should have some resize
            v.trace = CartesianRepresentation(x=np.zeros(len(tx), np.float64), y=np.zeros(len(tx), np.float64), z=np.zeros(len(tx), np.float64))
            v.trace.x = tx
            # v.trace.y = self.tefield.trace[i][1]
            # v.trace.z = self.tefield.trace[i][2]
            v.trace.y = trace[1]
            v.trace.z = trace[2]

            # Generate the time array
            v.calculate_t_vector(min_t0)

            v.du_id = self.tefield.du_id[i]

            self.efields.append(v)

        return ret

    ## Fill part of the event from the Shower tree
    def fill_event_from_shower_tree(self, simshower=False):
        ret = 1
        # The shower contains simulated parameters
        if simshower:
            # Initialise the Shower
            self.simshower = Shower()
            tree = self.tsimshower
            shower = self.simshower
        # The shower contains reconstructed parameters
        else:
            # Initialise the Shower
            self.shower = Shower()
            tree = self.tshower
            shower = self.shower

        if self._entry_number is not None:
            ret = tree.get_entry(self._entry_number)
        else:
            ret = tree.get_event(self.event_number, self.run_number)
        ## Shower primary particle type
        shower.primary_type = tree.primary_type
        ## Shower energy from e+- (ie related to radio emission) (GeV)
        shower.energy_em = tree.energy_em
        ## Shower total energy of the primary (including muons, neutrinos, ...) (GeV)
        shower.energy_primary = tree.energy_primary
        ## Shower Xmax [g/cm2]
        shower.Xmax = tree.xmax_grams
        ## Shower position in the site's reference frame
        shower.Xmaxpos = tree.xmax_pos_shc
        ## Shower azimuth
        shower.azimuth = tree.azimuth
        ## Shower zenith
        shower.zenith = tree.zenith
        ## Direction of the origin
        shower.origin_geoid = self.trun.origin_geoid
        ## Poistion of the core on the ground in the site's reference frame
        shower.core_ground_pos = tree.shower_core_pos
        ## Magnetic field in the place of shower
        shower.magnetic_field = tree.magnetic_field

        return ret

    ## Print all the class values
    def print(self):
        # Assign the TTree branches to the class fields
        for field in fields(self):
            # Skip the list fields
            if any(x in field.name for x in {"antennas", "voltages", "efields", "shower", "trun", "tvoltage", "tefield", "tshower"}): continue
            print("{:<30} {:>30}".format(field.name, str(getattr(self, field.name))))

        # Now deal with the list fields separately

        print("Shower:")
        print("\t{:<30} {:>30}".format("Energy EM:", self.shower.energy_em))
        print("\t{:<30} {:>30}".format("Xmax [g/cm2]:", self.shower.Xmax))
        print("\t{:<30} {:>30}".format("Xmax position:", str(self.shower.Xmaxpos.ravel())))
        print("\t{:<30} {:>30}".format("Origin geoid:", str(self.shower.origin_geoid.ravel())))
        print("\t{:<30} {:>30}".format("Core ground pos:", str(self.shower.core_ground_pos.ravel())))

        print("Antennas:")
        print("\t{:<30} {:>30}".format("No of antennas:", len(self.antennas)))
        print("\t{:<30} {:>30}".format("Position:", str([a.position.ravel() for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Tilt:", str([a.tilt.ravel() for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Acceleration:", str([a.acceleration.ravel() for a in self.antennas])))
        # print("\t{:<30} {:>30}".format("Humidity:", str([a.atm_humidity for a in self.antennas])))
        # print("\t{:<30} {:>30}".format("Pressure:", str([a.atm_pressure for a in self.antennas])))
        # print("\t{:<30} {:>30}".format("Temperature:", str([a.atm_temperature for a in self.antennas])))
        # print("\t{:<30} {:>30}".format("Battery level:", str([a.battery_level for a in self.antennas])))
        # print("\t{:<30} {:>30}".format("Firmware version:", str([a.firmware_version for a in self.antennas])))

        print("Voltages:")
        print("\t{:<30} {:>30}".format("Triggered status:", str([tr.is_triggered for tr in self.voltages])))
        print("\t{:<30} {:>30}".format("Traces lengths:", str([len(tr.trace[0]) for tr in self.voltages])))
        print("\t{:<30} {:>30}".format("Traces first values:", str([tr.trace[0][0] for tr in self.voltages])))

        print("Efields:")
        print("\t{:<30} {:>30}".format("Traces lengths:", str([len(tr.trace[0]) for tr in self.efields])))
        print("\t{:<30} {:>30}".format("Traces first values:", str([tr.trace[0][0] for tr in self.efields])))

    ## Write the Event to a file/directory
    def write(self, common_filename=None, shower_filename=None, efields_filename=None, voltages_filename=None, run_filename=None, overwrite=False, out_dir=None):

        # *** Writing to the current files (no output directory provided or same as current) ***

        if out_dir is None or (isinstance(out_dir, str) and self._directory.dir_name==out_dir) or (isinstance(out_dir, DataDirectory) and self._directory.dir_name==out_dir.dir_name):
            # Give common_filename to all the filenames if not specified
            if common_filename:
                if not shower_filename: shower_filename = common_filename
                if not efields_filename: efields_filename = common_filename
                if not voltages_filename: voltages_filename = common_filename
                if not run_filename: run_filename = common_filename

            # Invoke saving for each part
            self.write_shower(shower_filename)
            self.write_efields(efields_filename)
            self.write_voltages(voltages_filename)
            self.write_run(run_filename)

        # *** Output directory was given ***
        else:
            # target_dir = None
            if isinstance(out_dir, str):
                target_dir_path = Path(out_dir)

                # Delete the directory if overwrite requested
                if target_dir_path.is_dir() and overwrite:
                    shutil.rmtree(target_dir_path)

                # Create the target directory if it doesn't exist
                target_dir_path.mkdir(exist_ok=True)

                # Init the target DataDirectory
                target_dir = DataDirectory(out_dir)
            else:
                target_dir = out_dir

            if not isinstance(target_dir, DataDirectory):
                print("ERROR: out_dir must be of type DataDirectory or string")
                exit(1)

            # Go through all the run trees
            # ToDo: Add trunrawvoltage
            for source_tree in self._trees:
                if source_tree == self.tshower: continue
                # print("source_tree:", source_tree, self._trees)
                # Skip non-existing trees
                if not source_tree: continue

                # Check if the tree exists in the target directory
                source_tree_name = source_tree.tree_name
                if not getattr(target_dir, source_tree_name):
                    # Create the tree and its file
                    create_file_tree(target_dir, source_tree_name, source_tree)
                    self.files_creation_time = target_dir.cur_time_string

                # Get the target tree from the target directory
                target_tree = getattr(target_dir, source_tree_name)

                # For run trees, don't add the run if it is already in the target tree
                if source_tree in self._run_trees and target_tree.has_run(self.run_number): continue

                # For event trees, don't add the run,event if it is already in the target tree
                if source_tree in self._event_trees and target_tree.has_event(self.event_number, self.run_number): continue

                # Copy the current event
                target_tree.copy_contents(source_tree)
                # Fill the target tree
                target_tree.fill()

                # Build index
                # For run trees
                if source_tree in self._run_trees:
                    target_tree.build_index("run_number")
                else:
                    target_tree.build_index("run_number", "event_number")

                # Write the tree
                print("Writing", target_tree.tree_name)
                # target_tree._tree.GetCurrentFile().Write("", ROOT.TObject.kWriteDelete)
                target_tree.write(force_close_file=True)

            # Write the shower if it is created (externally)
            if self.shower:
                print("Writing shower")
                file_name = f"shower_{self.files_creation_time}_0-0_L1_0000.root"
                self.fill_shower_tree(filename=file_name, tree_name="tshower")
                self.write_shower(target_dir.dir_name + "/" + file_name)
                self.tshower.stop_using()

    ## Write the run to a file
    def write_run(self, filename, overwrite=False):
        self.fill_run_tree(filename=filename)
        if self.auto_file_close:
            self.trun.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)

    ## Write the voltages to a file
    def write_voltages(self, filename, overwrite=False):
        self.fill_voltage_tree(filename=filename)
        if self.auto_file_close:
            self.tvoltage.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)

    ## Write the efields to a file
    def write_efields(self, filename, overwrite=False):
        self.fill_efield_tree(filename=filename)
        if self.auto_file_close:
            self.tefield.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)

    ## Write the shower to a file
    def write_shower(self, filename, overwrite=False, tree_name="tshower"):
        self.fill_shower_tree(filename=filename, tree_name=tree_name)
        if self.auto_file_close:
            self.tshower.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)


    ## Fill the run tree from this Event
    def fill_run_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.trun is not None and not overwrite:
            raise TreeExists("The trun TTree already exists!")

        # Look for the TRun with the same file and name in the memory
        for el in grand_tree_list:
            # If the TRun with the same file and name in the memory exists, use it
            if type(el)==TRun and el._tree_name== "trun" and el._file_name==filename:
                self.trun = el
                break
        # No same TRun in memory - create a new one
        else:
            self.trun = TRun(_file_name=filename, _tree_name="trun")

        # Copy the event into the tree
        self.trun.run_number = self.run_number
        self.trun.run_mode = self.run_mode
        self.trun.data_source = self.data_source
        self.trun.data_generator = self.data_generator
        self.trun.data_generator_version = self.data_generator_version
        self.trun.site = self.site
        # self.trun.site_long = self.site_long
        # self.trun.site_lat = self.site_lat
        self.trun.origin_geoid = self.origin_geoid[:,0]
        self.trun.t_bin_size = self._t_bin_size

        # Fill the tree with values
        try:
            self.trun.fill()
        # If this Run already exists just don't fill
        except NotUniqueEvent:
            pass


    ## Fill the voltage tree from this Event
    def fill_voltage_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.tvoltage is not None and not overwrite:
            raise TreeExists("The tvoltage TTree already exists!")

        # Look for the TVoltage with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TVoltage with the same file and name in the memory exists, use it
            if type(el)==TVoltage and el._tree_name== "tvoltage" and el._file_name==filename:
                self.tvoltage = el
                break
        # No same TVoltage in memory - create a new one
        else:
            self.tvoltage = TVoltage(_file_name = filename)

        self.tvoltage.run_number = self.run_number
        self.tvoltage.event_number = self.event_number

        # Copy the contents of voltages to the tree

        # Set the DU id
        self.tvoltage.du_id = [v.du_id for v in self.voltages]

        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        # self.tvoltage.trace = [[np.array(v.trace.x).astype(np.float32), np.array(v.trace.y).astype(np.float32), np.array(v.trace.z).astype(np.float32)] for v in self.voltages]
        self.tvoltage.trace = [v.trace for v in self.voltages]
        # self.tvoltage.trace_x = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.voltages]

        # Fill the times from t0
        self.tvoltage.du_seconds = [v.t0.astype('datetime64[s]').astype(np.int64) for v in self.voltages]
        self.tvoltage.du_nanoseconds = [(v.t0.astype('datetime64[ns]').astype(np.int64)-v.t0.astype('datetime64[s]').astype(np.int64)*1e9).astype(np.int64) for v in self.voltages]

        # Copy the contents of antennas to the tree
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.tvoltage.atm_temperature = np.array([np.array(a.atm_temperature) for a in self.antennas])
        self.tvoltage.atm_pressure = np.array([np.array(a.atm_pressure) for a in self.antennas])
        self.tvoltage.atm_humidity = np.array([np.array(a.atm_humidity) for a in self.antennas])
        self.tvoltage.battery_level = np.array([np.array(a.battery_level) for a in self.antennas])
        self.tvoltage.firmware_version = np.array([np.array(a.firmware_version) for a in self.antennas])

        self.tvoltage.fill()

    ## Fill the efield tree from this Event
    def fill_efield_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.tefield is not None and not overwrite:
            raise TreeExists("The tefield TTree already exists!")

        # Look for the TEfield with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TEfield with the same file and name in the memory exists, use it
            if type(el)==TEfield and el._tree_name== "tefield" and el._file_name==filename:
                self.tefield = el
                break
        # No same TEfield in memory - create a new one
        else:
            self.tefield = TEfield(_file_name = filename)

        self.tefield.run_number = self.run_number
        self.tefield.event_number = self.event_number

        # Copy the contents of efields to the tree

        # Set the DU id
        self.tefield.du_id = [v.du_id for v in self.voltages]

        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        # self.tefield.trace = [[np.array(v.trace.x).astype(np.float32) for v in self.efields], [np.array(v.trace.y).astype(np.float32) for v in self.efields], [np.array(v.trace.z).astype(np.float32) for v in self.efields]]
        self.tefield.trace = [v.trace for v in self.efields]
        # self.tefield.trace_x = [np.array(v.trace.x).astype(np.float32) for v in self.efields]
        # self.tefield.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.efields]
        # self.tefield.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.efields]
        # self.tefield.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.efields]
        # self.tefield.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.efields]
        # self.tefield.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.efields]

        # Fill the times from t0
        self.tefield.du_seconds = [v.t0.astype('datetime64[s]').astype(np.int64) for v in self.efields]
        self.tefield.du_nanoseconds = [(v.t0.astype('datetime64[ns]').astype(np.int64)-v.t0.astype('datetime64[s]').astype(np.int64)*1e9).astype(np.int64) for v in self.efields]

        self.tefield.fill()

    ## Fill the shower tree from this Event
    def fill_shower_tree(self, overwrite=False, filename=None, tree_name="tshower"):
        # Fill only if the tree not initialised yet
        # if self.tshower is not None and not overwrite:
        #     raise TreeExists("The tshower TTree already exists!")

        # Look for the TShower with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TShower with the same file and name in the memory exists, use it
            if type(el)==TShower and el._tree_name== "tshower" and el._file_name==filename:
                self.tshower = el
                break
        # No same TShower in memory - create a new one
        else:
            self.tshower = TShower(_file_name=filename, _tree_name=tree_name)

        self.tshower.run_number = self.run_number
        self.tshower.event_number = self.event_number


        self.tshower.energy_em = self.shower.energy_em
        self.tshower.energy_primary = self.shower.energy_primary
        ## Shower Xmax [g/cm2]
        self.tshower.xmax_grams = self.shower.Xmax
        ## Shower position in the site's reference frame
        self.tshower.xmax_pos = self.shower.Xmaxpos[:,0]
        ## Shower azimuth
        self.tshower.azimuth = self.shower.azimuth
        ## Shower zenith
        self.tshower.zenith = self.shower.zenith
        ## Poistion of the core on the ground in the site's reference frame
        self.tshower.shower_core_pos = self.shower.core_ground_pos[:,0]

        ## The analysis level is usually 1
        self.tshower.analysis_level = 1

        self.tshower.fill()

    def close_files(self):
        """Close all files of the all trees - needed when auto_file_close is False"""
        self.tshower.write()
        self.tefield.write()
        self.tvoltage.write()
        self.trun.write()
        self.tshower.close_file()
        self.tefield.close_file()
        self.tvoltage.close_file()
        self.trun.close_file()

    def fill_t_vector(self, resolution=1):
        """Fills the event's time vector with resolution resolution"""

        # Get the filled traces
        filled_vals = [el for el in [self.voltages, self.efields] if el is not None][0]

        t_vectors = [el.t_vector for el in filled_vals]

        # For the same length traces, easy min/max finding with standard numpy array
        try:
            t_vectors = np.array(t_vectors)
            # Get the starting time from traces
            st = np.min(t_vectors)
            # Get the ending time from traces
            et = np.max(t_vectors)
        # Non-rectangular array -> array of objects -> double search for min/max (slower)
        except:
            t_vectors = [el.t_vector.tolist() for el in filled_vals]
            # Get the starting time from traces
            st = min(min(t_vectors))
            # Get the ending time from traces
            et = max(max(t_vectors))

        self.t_vector = np.arange((et-st)/resolution+1)*resolution+st

    def get_voltage_at_time(self, t):
        """Get the voltage signal value in all the DUs at the given time"""
        return np.array([el.get_value_at_time(t) for el in self.voltages])

    def get_efield_at_time(self, t):
        """Get the efield signal value in all the DUs at the given time"""
        return np.array([el.get_value_at_time(t) for el in self.efields])

    def get_hilbert_voltage_at_time(self, t):
        """Get the voltage signal value in all the DUs at the given time"""
        return np.array([el.get_hilbert_value_at_time(t) for el in self.voltages])

    def get_hilbert_efield_at_time(self, t):
        """Get the efield signal value in all the DUs at the given time"""
        return np.array([el.get_hilbert_value_at_time(t) for el in self.efields])


# Create the tree and its file
def create_file_tree(target_dir, tree_name, source_tree):

    # Check if the time string was already generated
    if not hasattr(target_dir, "cur_time_string"):
        # Generate the time string and store it
        from datetime import datetime
        setattr(target_dir, "cur_time_string", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Generate the file name

    # If run file
    if tree_name[:4]=="trun":
        # Replace the run number
        file_name = f"{tree_name[1:]}_00000_L{source_tree.analysis_level}_0000.root"
    else:
        # Replace the date and event numbers
        file_name = f"{tree_name[1:]}_{target_dir.cur_time_string}_0-0_L{source_tree.analysis_level}_0000.root"

    # Get the tree class for this tree type
    tree_class = getattr(grand.dataio, source_tree.type)

    # Create the tree instance
    tree_instance = tree_class(_tree_name=source_tree.tree_name, _file_name=target_dir.dir_name+"/"+file_name)

    # Copy/create some metadata
    tree_instance.analysis_level = source_tree.analysis_level
    tree_instance.modification_software = "extract_events.py"

    # Attach the tree instance to the DataDirectory
    setattr(target_dir, tree_name, tree_instance)
