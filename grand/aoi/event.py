# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field, fields

import numpy as np

from grand.basis import validate as _validate
import os
import ROOT
from pathlib import Path

from grand import CartesianRepresentation
from grand.aoi.timetrace import Voltage, Efield, TreeExists
from grand.aoi.antenna import Antenna
from grand.aoi.shower import Shower
from grand.dataio import DataDirectory, TRun, TRunRawVoltage, TVoltage, TEfield, TShower, TRawVoltage, grand_tree_list, NotUniqueEvent 
import grand.dataio
from grand.dataio.xmax_frame import xmax_above_ground

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        """Returns *func* unchanged, standing in for ``line_profiler.profile``.

        ``line_profiler`` is an optional development dependency, so the
        decorator has to keep working when it is absent.

        Parameters
        ----------
        func : callable
            The function that would have been profiled.

        Returns
        -------
        callable
            ``func`` itself, unmodified.
        """
        return func


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

    _all_antennas_key: tuple = None
    """How and for which site `_all_antennas` was built, as ``(source, site)``.

    The dict is filled either from GPS coordinates (``"gps"``) or from the run's
    own `du_xyz` (``"run"``), which are not interchangeable. Reusing it across a
    change of either would hand back positions derived the other way.
    """

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

    # Trees filled for writing with auto_file_close False, written by close_files()
    _pending_writes: list = None

    ## Post-init actions, like an automatic readout from files, etc.
    def __post_init__(self):
        # If the file name was given, init the Event from trees
        r"""Completes initialisation after the dataclass fields are set.

        """
        if self._file:
            self.fill_event_from_trees()
            self.fill_t_vector()

    @property
    def file(self):
        """A single file that contains all the TTrees

        Returns
        -------
        str
            Path of the file this event was read from or will be written to.
        """
        return str(self._file)

    @file.setter
    def file(self, value):
        """A single file that contains all the TTrees

        Parameters
        ----------
        value : str
            Path of the file this event is read from or written to.
        """

        # If the _file is not yet TFile, make it so; a path object or a file
        # ROOT cannot open used to give a bare cppyy error (#235)
        if not isinstance(value, ROOT.TFile):
            from grand.dataio.data_handling import _open_root_file
            self._file = _open_root_file(os.fspath(value))
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
        """A single file that contains all the TTrees

        Returns
        -------
        str
            Directory the event files live in.
        """
        return self._directory

    @directory.setter
    def directory(self, value):
        """A directory that contains all the files with TTrees

        Parameters
        ----------
        value : str
            Path of the directory the event files live in.
        """
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
        """Origin of the coordinate system used for the array

        Returns
        -------
        ndarray, shape (3,)
            Latitude, longitude and height of the array origin.
        """
        return self._origin_geoid

    @origin_geoid.setter
    def origin_geoid(self, v):
        r"""Sets the geodetic origin the event's local coordinates refer to.

        Parameters
        ----------
        v : array_like
            Latitude, longitude and height of the array origin.
        """
        self._origin_geoid = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    ## Fill this event from trees
    def fill_event_from_trees(self, event_number=None, run_number=None, entry_number=None, simshower=False, use_trawvoltage=False, trawvoltage_channels=[0,1,2], init_trees=True, gp300_workaround=True, tefield_level=None):
        """Fill this event from trees

        Parameters
        ----------
        event_number : int, optional
            Event number.
        run_number : int, optional
            Run number.
        entry_number : int, optional
            Entry index, instead of the pair above.
        simshower : bool, optional
            Read the simulator-only shower tree.
        use_trawvoltage : bool, optional
            Read raw voltages rather than calibrated ones.
        trawvoltage_channels : sequence, optional
            Channels to read.
        init_trees : bool, optional
            Open the trees before reading.
        gp300_workaround : bool, optional
            Apply the GP300 antenna-ordering workaround.
        tefield_level : int, optional
            Analysis level of the electric-field tree to read. The level is
            recorded in the tree's metadata rather than in its name, so
            choosing one means choosing a file and needs `directory` to be
            set. Without one, the level of the open file is checked against
            this. `None` reads whatever is there.

        Returns
        -------
        bool
            True when the event was found and populated.

        Raises
        ------
        ValueError
            When `tefield_level` names a level the data does not hold. Reading
            a different level than the one asked for would not be visible to
            the caller, so it is refused instead.
        """
        # Check if any of the files exist
        if not self._file and not self.file_trun and not self.file_trunrawvoltage and not self.file_tvoltage and not self.file_tefield and not self.file_tshower and not self.file_tsimshower:
            # Printed "Aborting." and returned False, which callers ignored,
            # leaving a half-initialised Event (#256)
            raise FileNotFoundError(_validate.message(
                "Event.fill_event_from_trees", "no file or directory to read the event from"))

        # *** Set the run/event/entry number if requested.

        # If entry/event/run number not specified, take the first entry
        run_entry_number = None
        if self._entry_number is None and self.run_number is None and self.event_number is None and entry_number is None and run_number is None and event_number is None:
            entry_number = 0
            run_entry_number = 0

        # Don't allow specifying entry and event/run at the same time, because... what to chose?
        if entry_number is not None and (run_number is not None or event_number is not None):
            raise ValueError(_validate.message(
                "Event.fill_event_from_trees",
                "give entry_number, or event_number and run_number, not both"))
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

        #if self.trecons is not None:
            # If initialising trees requested
        #    if init_trees:
                # Check the TRecons tree existence
        #        if trecons:= self.trecons.Get("trecons"):
        #            self.trecons = TRecons(_tree=trecons)
        #        else:
        #            print("No TRecons tree. Reconstructed event information will not be available.")
        #            self.trecons = None
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
            # With a directory the trees are opened once, so the voltage tree
            # must be chosen on every call: a use_trawvoltage=True call left
            # TRawVoltage in place, and the next default call failed on its
            # missing 'trace' (#213)
            if not init_trees and self.directory is not None:
                if use_trawvoltage and self.directory.ftrawvoltages:
                    self.tvoltage = self.directory.trawvoltage
                elif not use_trawvoltage and self.directory.ftvoltages:
                    self.tvoltage = self.directory.tvoltage
            # Raw voltages are all the data holds: read them as such rather
            # than fail on TRawVoltage's missing 'trace' (#213, #235)
            if not use_trawvoltage and isinstance(self.tvoltage, TRawVoltage):
                use_trawvoltage = True
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
            # The analysis level is not part of the tree name: every file
            # stores its tree as "tefield" and records the level in the
            # tree's UserInfo. Selecting a level therefore means selecting
            # a file, which only the DataDirectory knows how to do.  This is
            # done on every call: it used to happen only when the trees were
            # first opened, so a per-call level was ignored (#213).
            if tefield_level is not None and self.directory is not None:
                self.tefield = getattr(self.directory, f"tefield_l{tefield_level}", None)
                if self.tefield is None:
                    raise ValueError(_validate.message(
                        "Event", f"no Efield tree of analysis level {tefield_level} in "
                        f"{self.directory.dir_name}"))
                self.tefield_level = tefield_level
            # No level asked: the directory's default, not the last one asked for
            elif self.directory is not None and not init_trees:
                self.tefield = self.directory.tefield
            # If initialising trees requested
            elif init_trees:
                # Check the Efield tree existence
                if tefield := self.file_tefield.Get("tefield"):
                    self.tefield = TEfield(_tree=tefield)
                    # Without a directory there is no choice of file, so the
                    # level is whatever the open one holds. Read it back rather
                    # than assume it is the one that was asked for.
                    if tefield_level is not None:
                        if self.tefield.analysis_level != tefield_level:
                            raise ValueError(
                                f"The open Efield file holds analysis level "
                                f"{self.tefield.analysis_level}, not the requested "
                                f"{tefield_level}.")
                        self.tefield_level = tefield_level
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

        self.fill_antennas(gp300_workaround=gp300_workaround)

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
        r"""Populates the run-level fields from the run tree.

        Parameters
        ----------
        run_entry_number : int, optional
            Entry to read.

        Returns
        -------
        bool
            True when the entry was found.
        """
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
        r"""Populates the run-level voltage fields from the raw-voltage run tree.

        Parameters
        ----------
        run_entry_number : int, optional
            Entry to read.

        Returns
        -------
        bool
            True when the entry was found.
        """
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
        """Fill event's antennas

        Parameters
        ----------
        gp300_workaround : bool, optional
            Apply the GP300 antenna-ordering workaround.
        """
        self.antennas = []

        # For GP300 for now, get the GPS coordinates for each DU and calculate the x/y/z here
        if gp300_workaround and ("GP300" in self.site or "GP80" in self.site
                                 or "GP13" in self.site):

            # Get the tree we are using
            cur_tree = None
            if self.tefield is not None:
                cur_tree = self.tefield
            elif self.tvoltage is not None:
                cur_tree = self.tvoltage
            else:
                # Raising a plain string is itself a TypeError in Python 3
                raise ValueError(_validate.message(
                    "Event.fill_antennas", "cannot calculate the antenna positions: the event "
                    "has neither an efield nor a voltage tree"))

            # If this is the first time we calculate antennas positions, or
            # the ones we hold were not built from GPS for this same site
            if not self._all_antennas or self._all_antennas_key != ("gps", self.site):
                print("GP300 workaround: calculating all antennas positions")
                from grand import Geodetic, GRANDCS

                # Get the coordinates for all DUs from all events
                count = cur_tree.draw("du_id:gps_lat:gps_long:gps_alt", "", "goff")
                if count == -1:
                    raise RuntimeError(_validate.message(
                        "Event.fill_antennas", "cannot read the antenna GPS positions "
                        "(du_id, gps_lat, gps_long, gps_alt) from the ROOT file"))

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

                self._all_antennas_key = ("gps", self.site)

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
            # Antenna positions come from the run tree.  A file holding only
            # event trees (say one efield file given to EventList) has none:
            # the event then has no antennas, as other missing trees are
            # skipped, rather than failing on trun.du_id.
            if self.trun is None:
                print("No Run tree. Antenna positions will not be available.")
                return

            # Fill the antenna part. With neither tree there is nothing to say
            # which DUs took part, so the event simply has no antennas -- the
            # name used to be left unbound and the loop below raised.
            event_dus_indices = []
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

            self._all_antennas_key = ("run", self.site)



    ## Fill part of the event from the Voltage tree
    def fill_event_from_voltage_tree(self, use_trawvoltage=False, trawvoltage_channels=(0,1,2)):
        r"""Populates the voltage traces from the voltage tree.

        Parameters
        ----------
        use_trawvoltage : bool, optional
            Read ``TRawVoltage`` rather than ``TVoltage``.
        trawvoltage_channels : sequence, optional
            Which channels to read; all of them by default.

        Returns
        -------
        bool
            True when traces were found.
        """
        # A voltage has three components, so three channels are read; fewer
        # crashed with IndexError (#213)
        if use_trawvoltage:
            channels = list(trawvoltage_channels)
            if len(channels) != 3 or not all(isinstance(c, (int, np.integer)) and not isinstance(c, bool)
                                             for c in channels):
                raise ValueError(_validate.message(
                    "Event", "'trawvoltage_channels' must name exactly 3 channels (the x, y "
                    "and z of the voltage), got %r" % (trawvoltage_channels,)))
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
            if not use_trawvoltage:
                trace = self.tvoltage.trace[i]
                tx = trace[0]
            else:
                trace = self.tvoltage.trace_ch[i]
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
            else:
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
        r"""Populates this event from the electric-field tree.

        Returns
        -------
        bool
            True when the tree held data for this event.
        """
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
            tx = trace[0]
            v.n_points = len(tx)
            v.t0 = np.datetime64(self.tefield.du_seconds[i] * 1000000000 + self.tefield.du_nanoseconds[i], "ns")
            # The default size of the CartesianRepresentation is wrong. ToDo: it should have some resize
            v.trace = CartesianRepresentation(x=np.zeros(len(tx), np.float64), y=np.zeros(len(tx), np.float64), z=np.zeros(len(tx), np.float64))
            v.trace.x = tx
            v.trace.y = trace[1]
            v.trace.z = trace[2]

            # Generate the time array
            v.calculate_t_vector(min_t0)

            v.du_id = self.tefield.du_id[i]

            self.efields.append(v)

        return ret

    ## Fill part of the event from the Shower tree
    def fill_event_from_shower_tree(self, simshower=False):
        r"""Populates the shower parameters from the shower tree.

        Parameters
        ----------
        simshower : bool, optional
            Read ``TShowerSim``, the simulator-only tree, rather than
            ``TShower``.

        Returns
        -------
        bool
            True when the shower was found.
        """
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
        # The ground altitude comes from the run tree; without one this failed
        # on "'NoneType' object has no attribute 'origin_geoid'" (#235)
        if self.trun is None:
            raise FileNotFoundError(_validate.message(
                "Event", "the shower needs the run tree (the site's origin) and this input has "
                "none: give the directory holding the run_*.root file, or a file with a trun "
                "tree"))
        # Above the ground whichever frame the file used (grand-mother/grand#160).
        shower.Xmaxpos, _ = xmax_above_ground(
            tree.xmax_pos_shc, tree.zenith, tree.azimuth,
            self.trun.origin_geoid[2])
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
        r"""Prints a summary of the event, for interactive use.

        """
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

        r"""Writes the whole event out, one file per tree kind.

        Parameters
        ----------
        common_filename : str, optional
            Single destination for every tree.  When given, the per-tree
            names below are ignored.
        shower_filename, efields_filename, voltages_filename, run_filename : str, optional
            Individual destinations for each tree.
        overwrite : bool, optional
            Replace what this event's trees would add to: with file names, the
            trees of those names in those files; with ``out_dir``, the files of
            those tree kinds in that directory.  Nothing else is removed.
            Without it, the event is added to what is there.
        out_dir : str, optional
            Directory to write into.

        Raises
        ------
        ValueError
            If neither ``out_dir`` nor a file name is given: writing back into
            the files the event was read from is not supported.

        Notes
        -----
        Writing never changes the trees this event was read from, so the event,
        and the ``EventList`` it came from, can still be used afterwards.
        Only the trees with a file name, explicit or through
        ``common_filename``, are written.
        """
        if out_dir is None or (isinstance(out_dir, str) and self._directory and self._directory.dir_name==out_dir) or (isinstance(out_dir, DataDirectory) and self._directory and self._directory.dir_name==out_dir.dir_name):
            # Give common_filename to all the filenames if not specified
            if common_filename:
                if not shower_filename: shower_filename = common_filename
                if not efields_filename: efields_filename = common_filename
                if not voltages_filename: voltages_filename = common_filename
                if not run_filename: run_filename = common_filename

            # Writing back into the source files was documented but crashed
            # (AttributeError on a None tree, #212); it is refused clearly
            if not any((shower_filename, efields_filename, voltages_filename, run_filename)):
                raise ValueError(_validate.message(
                    "Event.write", "give out_dir, common_filename or a file name per tree; "
                    "writing back into the files the event was read from is not supported"))

            # Invoke saving for each part, passing overwrite on (it was
            # dropped, so common_filename always failed with TreeExists, #212)
            # Parts the event does not hold are skipped rather than crashing
            if shower_filename and self.shower is not None: self.write_shower(shower_filename, overwrite=overwrite)
            if efields_filename and self.efields: self.write_efields(efields_filename, overwrite=overwrite)
            if voltages_filename and self.voltages: self.write_voltages(voltages_filename, overwrite=overwrite)
            if run_filename: self.write_run(run_filename, overwrite=overwrite)

        # *** Output directory was given ***
        else:
            # target_dir = None
            if isinstance(out_dir, str):
                target_dir_path = Path(out_dir)

                # Replace only the files this write produces: this removed the
                # whole directory, with whatever else the user kept in it (#212)
                if target_dir_path.is_dir() and overwrite:
                    for source_tree in self._trees or []:
                        if source_tree:
                            for old in target_dir_path.glob(source_tree.tree_name[1:] + "_*.root"):
                                old.unlink()

                # Create the target directory if it doesn't exist
                target_dir_path.mkdir(exist_ok=True)

                # Init the target DataDirectory
                target_dir = DataDirectory(out_dir)
            else:
                target_dir = out_dir

            if not isinstance(target_dir, DataDirectory):
                raise TypeError(_validate.message(
                    "Event.write", "'out_dir' must be a directory name or a DataDirectory, got %s"
                    % type(target_dir).__name__))

            # Go through all the run trees
            # ToDo: Add trunrawvoltage
            for source_tree in self._trees:
                # print("source_tree:", source_tree, self._trees)
                # Skip non-existing trees
                if not source_tree: continue

                # Check if the tree exists in the target directory
                source_tree_name = source_tree.tree_name
                if not getattr(target_dir, source_tree_name):
                    # Create the tree and its file
                    create_file_tree(target_dir, source_tree_name, source_tree)

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

    ## Write the run to a file
    def write_run(self, filename, overwrite=False):
        r"""Writes the run tree to a ROOT file.

        Parameters
        ----------
        filename : str, optional
            Destination file.
        overwrite : bool, optional
            Replace the tree of this kind in that file rather than adding to it.
        """
        tree_name_ = 'trun'
        if overwrite:
            _drop_tree(filename, tree_name_)
        tree = self._make_run_tree(filename=filename)
        self._finish_write(tree, filename)

    ## Write the voltages to a file
    def write_voltages(self, filename, overwrite=False):
        r"""Writes the voltage traces to a ROOT file.

        Parameters
        ----------
        filename : str, optional
            Destination file.
        overwrite : bool, optional
            Replace the tree of this kind in that file rather than adding to it.
        """
        tree_name_ = 'tvoltage'
        if overwrite:
            _drop_tree(filename, tree_name_)
        tree = self._make_voltage_tree(filename=filename)
        self._finish_write(tree, filename)

    ## Write the efields to a file
    def write_efields(self, filename, overwrite=False):
        r"""Writes the electric-field traces to a ROOT file.

        Parameters
        ----------
        filename : str, optional
            Destination file.
        overwrite : bool, optional
            Replace the tree of this kind in that file rather than adding to it.
        """
        tree_name_ = 'tefield'
        if overwrite:
            _drop_tree(filename, tree_name_)
        tree = self._make_efield_tree(filename=filename)
        self._finish_write(tree, filename)

    ## Write the shower to a file
    def write_shower(self, filename, overwrite=False, tree_name="tshower"):
        r"""Writes the shower parameters to a ROOT file.

        Parameters
        ----------
        filename : str, optional
            Destination file.
        overwrite : bool, optional
            Replace the tree of this kind in that file rather than adding to it.
        tree_name : str, optional
            Name of the tree to write, which selects ``TShower`` or
            ``TShowerSim``.
        """
        tree_name_ = tree_name
        if overwrite:
            _drop_tree(filename, tree_name_)
        tree = self._make_shower_tree(filename=filename, tree_name=tree_name)
        self._finish_write(tree, filename)


    ## Fill the run tree from this Event
    def fill_run_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        r"""Fills the run tree from this event's contents, ready to be written.

        Parameters
        ----------
        overwrite : bool, optional
            Replace existing entries rather than appending.
        filename : str, optional
            File the tree belongs to.
        """
        if self.trun is not None and not overwrite:
            raise TreeExists("The trun TTree already exists!")
        self.trun = self._make_run_tree(filename=filename)

    def _make_run_tree(self, filename=None):
        r"""Builds and fills a TRun for writing, without attaching it to the event.

        Parameters
        ----------
        filename : str, optional
            File the tree belongs to.

        Returns
        -------
        TRun
            The filled tree.
        """

        # Look for the TRun with the same file and name in the memory
        for el in grand_tree_list:
            # If the TRun with the same file and name in the memory exists, use it
            if type(el)==TRun and el._tree_name== "trun" and el._file_name==filename:
                tree = el
                break
        # No same TRun in memory - create a new one
        else:
            tree = TRun(_file_name=filename, _tree_name="trun")

        # Copy the event into the tree
        tree.run_number = self.run_number
        tree.run_mode = self.run_mode
        tree.data_source = self.data_source
        tree.data_generator = self.data_generator
        tree.data_generator_version = self.data_generator_version
        tree.site = self.site
        # tree.site_long = self.site_long
        # tree.site_lat = self.site_lat
        tree.origin_geoid = self.origin_geoid[:,0]
        # Read events hold one sampling time; the branch is a vector (#212)
        tree.t_bin_size = np.atleast_1d(self._t_bin_size)

        # Fill the tree with values
        try:
            tree.fill()
        # If this Run already exists just don't fill
        except NotUniqueEvent:
            pass
        return tree

    ## Fill the voltage tree from this Event
    def fill_voltage_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        r"""Fills the voltage tree from this event's contents, ready to be written.

        Parameters
        ----------
        overwrite : bool, optional
            Replace existing entries rather than appending.
        filename : str, optional
            File the tree belongs to.
        """
        if self.tvoltage is not None and not overwrite:
            raise TreeExists("The tvoltage TTree already exists!")
        self.tvoltage = self._make_voltage_tree(filename=filename)

    def _make_voltage_tree(self, filename=None):
        r"""Builds and fills a TVoltage for writing, without attaching it to the event.

        Parameters
        ----------
        filename : str, optional
            File the tree belongs to.

        Returns
        -------
        TVoltage
            The filled tree.
        """

        # Look for the TVoltage with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TVoltage with the same file and name in the memory exists, use it
            if type(el)==TVoltage and el._tree_name== "tvoltage" and el._file_name==filename:
                tree = el
                break
        # No same TVoltage in memory - create a new one
        else:
            tree = TVoltage(_file_name = filename)

        tree.run_number = self.run_number
        tree.event_number = self.event_number

        # Copy the contents of voltages to the tree

        # Set the DU id
        tree.du_id = [v.du_id for v in self.voltages]

        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        # tree.trace = [[np.array(v.trace.x).astype(np.float32), np.array(v.trace.y).astype(np.float32), np.array(v.trace.z).astype(np.float32)] for v in self.voltages]
        tree.trace = [v.trace for v in self.voltages]
        # tree.trace_x = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # tree.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # tree.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.voltages]
        # tree.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.voltages]
        # tree.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.voltages]
        # tree.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.voltages]

        # Fill the times from t0
        tree.du_seconds = [v.t0.astype('datetime64[s]').astype(np.int64) for v in self.voltages]
        tree.du_nanoseconds = [(v.t0.astype('datetime64[ns]').astype(np.int64)-v.t0.astype('datetime64[s]').astype(np.int64)*1e9).astype(np.int64) for v in self.voltages]

        # The antennas' monitoring fields (atm_*, battery_level,
        # firmware_version) are not written: TVoltage has none of them, and
        # assigning them stored nothing, and now raises (#202)

        tree.fill()
        return tree

    ## Fill the efield tree from this Event
    def fill_efield_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        r"""Fills the electric-field tree from this event's contents, ready to be written.

        Parameters
        ----------
        overwrite : bool, optional
            Replace existing entries rather than appending.
        filename : str, optional
            File the tree belongs to.
        """
        if self.tefield is not None and not overwrite:
            raise TreeExists("The tefield TTree already exists!")
        self.tefield = self._make_efield_tree(filename=filename)

    def _make_efield_tree(self, filename=None):
        r"""Builds and fills a TEfield for writing, without attaching it to the event.

        Parameters
        ----------
        filename : str, optional
            File the tree belongs to.

        Returns
        -------
        TEfield
            The filled tree.
        """

        # Look for the TEfield with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TEfield with the same file and name in the memory exists, use it
            if type(el)==TEfield and el._tree_name== "tefield" and el._file_name==filename:
                tree = el
                break
        # No same TEfield in memory - create a new one
        else:
            tree = TEfield(_file_name = filename)

        tree.run_number = self.run_number
        tree.event_number = self.event_number

        # Copy the contents of efields to the tree

        # Set the DU id
        tree.du_id = [v.du_id for v in self.voltages]

        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        # tree.trace = [[np.array(v.trace.x).astype(np.float32) for v in self.efields], [np.array(v.trace.y).astype(np.float32) for v in self.efields], [np.array(v.trace.z).astype(np.float32) for v in self.efields]]
        tree.trace = [v.trace for v in self.efields]
        # tree.trace_x = [np.array(v.trace.x).astype(np.float32) for v in self.efields]
        # tree.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.efields]
        # tree.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.efields]
        # tree.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.efields]
        # tree.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.efields]
        # tree.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.efields]

        # Fill the times from t0
        tree.du_seconds = [v.t0.astype('datetime64[s]').astype(np.int64) for v in self.efields]
        tree.du_nanoseconds = [(v.t0.astype('datetime64[ns]').astype(np.int64)-v.t0.astype('datetime64[s]').astype(np.int64)*1e9).astype(np.int64) for v in self.efields]

        tree.fill()
        return tree

    ## Fill the shower tree from this Event
    def fill_shower_tree(self, overwrite=False, filename=None, tree_name="tshower"):
        # Fill only if the tree not initialised yet
        r"""Fills the shower tree from this event's contents.

        Parameters
        ----------
        overwrite : bool, optional
            Replace existing entries.
        filename : str, optional
            File the tree belongs to.
        tree_name : str, optional
            Which shower tree to fill.
        """
        if self.tshower is not None and not overwrite:
            raise TreeExists("The tshower TTree already exists!")
        self.tshower = self._make_shower_tree(filename=filename, tree_name=tree_name)

    def _make_shower_tree(self, filename=None, tree_name="tshower"):
        r"""Builds and fills a TShower for writing, without attaching it to the event.

        Parameters
        ----------
        filename : str, optional
            File the tree belongs to.
        tree_name : str, optional
            Which shower tree to fill.

        Returns
        -------
        TShower
            The filled tree.
        """

        # Look for the TShower with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the TShower with the same file and name in the memory exists, use it
            if type(el)==TShower and el._tree_name== "tshower" and el._file_name==filename:
                tree = el
                break
        # No same TShower in memory - create a new one
        else:
            tree = TShower(_file_name=filename, _tree_name=tree_name)

        tree.run_number = self.run_number
        tree.event_number = self.event_number


        tree.energy_em = self.shower.energy_em
        tree.energy_primary = self.shower.energy_primary
        ## Shower Xmax [g/cm2]
        tree.xmax_grams = self.shower.Xmax
        ## Xmax relative to the shower core, above the ground: what the
        ## readers (this class included) read back
        tree.xmax_pos_shc = self.shower.Xmaxpos[:,0]
        ## Xmax in the site's reference frame: the same point plus the core,
        ## as sim2root writes it (#104)
        tree.xmax_pos = self.shower.Xmaxpos[:,0] + self.shower.core_ground_pos[:,0]
        ## Shower azimuth
        tree.azimuth = self.shower.azimuth
        ## Shower zenith
        tree.zenith = self.shower.zenith
        ## Poistion of the core on the ground in the site's reference frame
        tree.shower_core_pos = self.shower.core_ground_pos[:,0]

        tree.fill()
        return tree

    def _finish_write(self, tree, filename):
        r"""Writes ``tree`` now, or keeps it for ``close_files()``.

        Parameters
        ----------
        tree : DataTree
            A tree built by one of the ``_make_*_tree`` methods.
        filename : str
            Its file.
        """
        if self.auto_file_close:
            tree.write(filename, force_close_file=True)
            tree.stop_using()
        else:
            if self._pending_writes is None:
                self._pending_writes = []
            if all(tree is not other for other in self._pending_writes):
                self._pending_writes.append(tree)

    def close_files(self):
        """Writes and closes the files of the trees written with auto_file_close False.

        Only trees this event filled for writing are written: the trees it was
        read from are left untouched (this used to write them too, changing
        the input files, #234).
        """
        for tree in self._pending_writes or []:
            # The same tree may be shared with other events writing to the file
            if tree.tree is None:
                continue
            tree.write(force_close_file=True)
            tree.stop_using()
        self._pending_writes = []

    def fill_t_vector(self, resolution=1):
        """Fills the event's time vector with resolution resolution

        Parameters
        ----------
        resolution : float, optional
            Time step of the common axis, in nanoseconds.
        """

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
        """Get the voltage signal value in all the DUs at the given time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray
            Voltage of each unit at that time.
        """
        return np.array([el.get_value_at_time(t) for el in self.voltages])

    def get_efield_at_time(self, t):
        """Get the efield signal value in all the DUs at the given time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray
            Electric field of each unit at that time.
        """
        return np.array([el.get_value_at_time(t) for el in self.efields])

    def get_hilbert_voltage_at_time(self, t):
        """Get the voltage signal value in all the DUs at the given time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray
            Hilbert envelope of the voltage at that time.
        """
        return np.array([el.get_hilbert_value_at_time(t) for el in self.voltages])

    def get_hilbert_efield_at_time(self, t):
        """Get the efield signal value in all the DUs at the given time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray
            Hilbert envelope of the field at that time.
        """
        return np.array([el.get_hilbert_value_at_time(t) for el in self.efields])


# Create the tree and its file
def _drop_tree(filename, tree_name):
    r"""Removes ``tree_name`` from ``filename``, if the file holds it.

    Parameters
    ----------
    filename : str
        The ROOT file.
    tree_name : str
        Name of the tree to remove.
    """
    if not filename or not os.path.isfile(filename):
        return
    from grand.dataio import file_lock

    file_lock.lock_for_writing(filename, "Event.write", fresh=True)
    try:
        f = ROOT.TFile(filename, "update")
        if f.GetListOfKeys().FindObject(tree_name):
            f.Delete(tree_name + ";*")
        f.Close()
    finally:
        file_lock.release(filename)


def create_file_tree(target_dir, tree_name, source_tree):

    # Check if the time string was already generated
    r"""Returns a tree of `tree_name` in `target_dir`, copying `source_tree` if given.

    Parameters
    ----------
    target_dir : str
    Directory the file lives in.
    tree_name : str
    Name of the tree to create.
    source_tree : DataTree, optional
    Tree whose structure and metadata to copy.

    Returns
    -------
    DataTree
        The new tree.
    """
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
