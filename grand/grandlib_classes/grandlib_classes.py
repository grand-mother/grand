## The grandlib classes following https://docs.google.com/document/d/1P0AwR3U3MVZyU1ewIobWkJPZmVkxKCAw/edit

from dataclasses import dataclass, field, fields
import numpy as np
from typing import Any
from grand.dataio.root_trees import *
from grand.geo.coordinates import *
import ROOT

@dataclass
class Antenna:
    """A class describing a single antenna"""

    id: int = -1
    """Antenna ID - the du_id from the trees"""

    ## Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)
    # position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    _position: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Antenna tilt
    _tilt: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Antenna acceleration - this comes from hardware. ToDo: perhaps recalculate to tilt or remove tilt?
    _acceleration: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))

    model: Any = 0
    """The antenna model"""

    # # ToDo: Parameters below come from the hardware, but do we want them here?
    # ## Atmospheric temperature (read via I2C)
    # atm_temperature: float = 0
    # ## Atmospheric pressure
    # atm_pressure: float = 0
    # ## Atmospheric humidity
    # atm_humidity: float = 0
    # ## Battery voltage
    # battery_level: float = 0
    # ## Firmware version
    # firmware_version: float = 0

    @property
    def position(self):
        """Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)"""
        return self._position

    @position.setter
    def position(self, v):
        self._position = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def tilt(self):
        """Antenna tilt"""
        return self._tilt

    @tilt.setter
    def tilt(self, v):
        self._tilt = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def acceleration(self):
        """Antenna acceleration - this comes from hardware."""
        return self._acceleration

    @acceleration.setter
    def acceleration(self, v):
        self._acceleration = CartesianRepresentation(x=v[0], y=v[1], z=v[2])


@dataclass
class Timetrace3D:
    """A class for holding x,y,z single antenna traces over time"""

    n_points: int = 0
    """The trace length"""

    time_step: float = 0
    """[ns] n_points x step = total timetrace length"""

    t0: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))
    """Start time of the trace as unix time with nanoseconds"""

    trigger_time: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))
    """Trigger time as unix time with nanoseconds"""

    t_bin_size: float = 2
    """The size of the time bin - the time resolution in ns"""

    du_id: int = -1
    """The Detector Unit ID"""

    ## Trace vector in X
    # trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Y
    # trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Z
    # trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    # ToDo: Allow empty constructor in CartesianRepresentation?
    ## Trace 3D vector (x,y,z)
    _trace: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), z=np.zeros(0, np.float32)))
    # _trace1: list = None
    # trace: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    t_vector: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    """The time vector [ns] - generated from the trace length, t0 and t_bin_size"""

    ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
    _hilbert_trace: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), z=np.zeros(0, np.float32)))
    # ## Hilbert envelope vector in X
    # hilbert_trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    # ## Hilbert envelope vector in X
    # hilbert_trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    # ## Hilbert envelope vector in X
    # hilbert_trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    ## ToDo: add additional quantities from the doc?

    ## ToDo: add additional quantities from the trees?

    def calculate_t_vector(self, time_offset):
        """Calculation of the time vector - should be called manually when all the necessary parameters of the Timetrace3D are set"""
        # ToDo: t0 is at the moment the trigger time, not the start time...
        self.t_vector = np.arange(self.trace.x.size)*self.t_bin_size+(self.t0-time_offset).astype(int)

    @property
    def trace(self):
        """Trace 3D vector (x,y,z)"""
        return self._trace

    @trace.setter
    def trace(self, v):
        self._trace = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def hilbert_trace(self):
        """Hilbert envelope 3D vector (x,y,z) - not defined in the hardware"""
        return self._hilbert_trace

    @hilbert_trace.setter
    def hilbert_trace(self, v):
        self._hilbert_trace = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

@dataclass
class Voltage(Timetrace3D):
    """A class for holding voltage traces + additional information"""

    ## GPS time of the trigger - why would we want it? We have already _trigger_time in Timetrace3D, that is GPS time + nanoseconds
    # _GPS_trigtime: np.uint32 = 0
    is_triggered: bool = True
    """Is this a triggered trace? - not sure if it should be here or in Timetrace3D, or perhaps further up in the event"""

@dataclass
class Efield(Timetrace3D):
    """A class for holding Efield traces + additional information"""

    eta: float = 0
    """Polarization angle of the reconstructed Efield in the shower plane [deg]"""

    a_ratio: float = 0
    """Ratio of the geomagnetic to charge excess contributions"""

@dataclass
class Shower:
    """A class for holding a shower"""

    energy_em: float = 0
    """Shower from e+- (ie related to radio emission) (GeV)"""

    energy_primary: float = 0
    """Total energy of the primary (including muons, neutrinos, ...) (GeV)"""

    Xmax: float = 0
    """Shower Xmax [g/cm2]"""

    _Xmaxpos: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    """Shower position in the site's reference frame"""

    azimuth: float = 0
    """Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")"""

    zenith: float = 0
    """Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")"""

    ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
    _origin_geoid: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Position of the core on the ground in the site's reference frame
    _core_ground_pos: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))

    @property
    def Xmaxpos(self):
        """Shower position in the site's reference frame"""
        return self._Xmaxpos

    @Xmaxpos.setter
    def Xmaxpos(self, v):
        self._Xmaxpos = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def origin_geoid(self):
        """Direction of origin"""
        return self._origin_geoid

    @origin_geoid.setter
    def origin_geoid(self, v):
        self._origin_geoid = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def core_ground_pos(self):
        """Position of the core on the ground in the site's reference frame"""
        return self._core_ground_pos

    @core_ground_pos.setter
    def core_ground_pos(self, v):
        self._core_ground_pos = CartesianRepresentation(x=v[0], y=v[1], z=v[2])



@dataclass
class Event:
    """A class for holding an event"""

    # ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    file: ROOT.TFile = None
    """The instance of the file with TTrees containing the event. """

    event_number: int = None
    """The current event in the file number"""

    run_number: int = None
    """The run number of the current event"""

    _entry_number: int = None
    """The entry number - used for enforcing loading specific entry from all the event trees. Makes sense only if those trees have all the same events."""

    # antennas: list[Antenna] = None
    antennas: list = None
    """Antennas participating in the event"""

    # voltages: list[Voltage] = None
    voltages: list = None
    """Voltages from different antennas"""

    # efields: list[Efield] = None
    efields: list = None
    """Efields from different antennas"""

    shower: Shower() = None
    """Reconstructed shower"""

    simshower: Shower() = None
    """Simualted shower for simulations"""

    ## ToDo: what is it?
    L: int = 0
    """Event multiplicity"""

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

    tvoltage: TVoltage = None
    """DOI's TVoltage/TRawVoltage tree containing voltage information"""

    tefield: TEfield = None
    """DOI's TEfield tree containing Efield information"""

    tshower: TShower = None
    """DOI's TShower tree containing reconstructed shower information"""

    tsimshower: TShower = None
    """DOI's TShower tree containing simulated shower information"""

    # Options

    auto_file_close: bool = True
    """Close files automatically after event write? - slower writing but less maitanance by the user"""

    ## Post-init actions, like an automatic readout from files, etc.
    def __post_init__(self):
        # If the file name was given, init the Event from trees
        if self.file:
            self.fill_event_from_trees()

    @property
    def origin_geoid(self):
        """Origin of the coordinate system used for the array"""
        return self._origin_geoid

    @origin_geoid.setter
    def origin_geoid(self, v):
        self._origin_geoid = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    ## Fill this event from trees
    def fill_event_from_trees(self, event_number=None, run_number=None, entry_number=None, simshower=False, use_trawvoltage=False, trawvoltage_channels=[0,1,2]):
        """Fill this event from trees
        :param simshower: how to treat the TShower existing in the file, as sim values  or reconstructed values
        :type simshower: bool
        """
        # Check if the file exist
        if not self.file:
            print("No file provided to init from. Aborting.")
            return False

        # If the _file is not yet TFile, make it so
        if not isinstance(self.file, ROOT.TFile):
            self.file = ROOT.TFile(self.file, "read")

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

        # *** Check what TTrees are available and fill according to their availability

        # Check the Run tree existence
        if trun := self.file.Get("trun"):
            self.trun = TRun(_tree=trun)
            # Fill part of the event from trun
            ret = self.fill_event_from_runtree(run_entry_number=run_entry_number)
            if ret: print("Run information loaded.")
            else: print("No Run tree. Run information will not be available.")
        else:
            print("No Run tree. Run information will not be available.")
            # Make trun really None
            self.trun = None

        # Use standard voltage tree
        if not use_trawvoltage:
            # Check the Voltage tree existence
            if tvoltage := self.file.Get("tvoltage"):
                self.tvoltage = TVoltage(_tree=tvoltage)
                # Fill part of the event from tvoltage
                ret = self.fill_event_from_voltage_tree()
                if ret: print("Voltage information loaded.")
                else:
                    print("No Voltage tree. Voltage information will not be available.")
                    # Make tvoltage really None
                    self.tvoltage = None
            else:
                print("No Voltage tree. Voltage information will not be available.")
                # Make tvoltage really None
                self.tvoltage = None
        # Use trawvoltage tree
        else:
            # Check the Voltage tree existence
            if tvoltage := self.file.Get("trawvoltage"):
                self.tvoltage = TRawVoltage(_tree=tvoltage)
                # Fill part of the event from tvoltage
                ret = self.fill_event_from_voltage_tree(use_trawvoltage=use_trawvoltage, trawvoltage_channels=trawvoltage_channels)
                if ret: print("Voltage information (from TRawVoltage) loaded.")
                else:
                    print("No TRawVoltage tree. Voltage information will not be available.")
                    # Make tvoltage really None
                    self.tvoltage = None
            else:
                print("No TRawVoltage tree. Voltage information will not be available.")
                # Make tvoltage really None
                self.tvoltage = None


        # Check the Efield tree existence
        if tefield := self.file.Get("tefield"):
            self.tefield = TEfield(_tree=tefield)
            # Fill part of the event from tefield
            ret = self.fill_event_from_efield_tree()
            if ret: print("Efield information loaded.")
            else:
                print("No Efield tree. Efield information will not be available.")
                # Make tefield really None
                self.tefield = None
        else:
            print("No Efield tree. Efield information will not be available.")
            # Make tefield really None
            self.tefield = None

        # Check the Shower tree existence
        if tshower := self.file.Get("tshower"):
            if simshower:
                self.tsimshower = TShower(_tree=tshower)
            else:
                self.tshower = TShower(_tree=tshower)
            # Fill part of the event from tshower
            ret = self.fill_event_from_shower_tree(simshower)
            if ret: print("Shower information loaded.")
            else:
                print("No Shower tree. Shower information will not be available.")
                # Make tshower really None
                if simshower:
                    self.tsimshower = None
                else:
                    self.tshower = None
        else:
            print("No Shower tree. Shower information will not be available.")
            # Make tshower really None
            if simshower:
                self.tsimshower = None
            else:
                self.tshower = None


    ## Fill part of the event from the Run tree
    def fill_event_from_runtree(self, run_entry_number=None):
        ret = 1

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
        self._t_bin_size = self.trun.t_bin_size

        self.antennas = []

        # Fill the antenna part
        for i in range(len(self.trun.du_id)):
            a = Antenna()
            a.id = self.trun.du_id[i]
            a.position.x = self.trun.du_xyz[i][0]
            a.position.y = self.trun.du_xyz[i][1]
            a.position.z = self.trun.du_xyz[i][2]
            a.tilt.x = self.trun.du_tilt[i][0]
            a.tilt.y = self.trun.du_tilt[i][1]

            self.antennas.append(a)

        return ret

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
            if not use_trawvoltage:
                tx = self.tvoltage.trace[i][0]
            else:
                tx = self.tvoltage.trace_ch[i][trawvoltage_channels[0]]
            v.n_points = len(tx)
            v.t0 = np.datetime64(self.tvoltage.du_seconds[i]*1000000000+self.tvoltage.du_nanoseconds[i], "ns")
            v.t_bin_size = self._t_bin_size
            # The default size of the CartesianRepresentation is wrong. ToDo: it should have some resize
            v.trace = CartesianRepresentation(x=np.zeros(len(tx), np.float64), y=np.zeros(len(tx), np.float64), z=np.zeros(len(tx), np.float64))
            v.trace.x = tx
            if not use_trawvoltage:
                v.trace.y = self.tvoltage.trace[i][1]
                v.trace.z = self.tvoltage.trace[i][2]
            else:
                v.trace.y = self.tvoltage.trace_ch[i][trawvoltage_channels[1]]
                v.trace.z = self.tvoltage.trace_ch[i][trawvoltage_channels[2]]

            # Generate the time array
            v.calculate_t_vector(min_t0)

            v.du_id = self.tvoltage.du_id[i]

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
        # Loop through traces
        for i in range(len(self.tefield.trace[0])):
            v = Efield()
            tx = self.tefield.trace[0][i]
            v.n_points = len(tx)
            v.t0 = np.datetime64(self.tefield.du_seconds[i] * 1000000000 + self.tefield.du_nanoseconds[i], "ns")
            # The default size of the CartesianRepresentation is wrong. ToDo: it should have some resize
            v.trace = CartesianRepresentation(x=np.zeros(len(tx), np.float64), y=np.zeros(len(tx), np.float64), z=np.zeros(len(tx), np.float64))
            v.trace.x = tx
            v.trace.y = self.tefield.trace[0][i]
            v.trace.z = self.tefield.trace[0][i]

            v.du_id = self.tvoltage.du_id[i]

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

    ## Write the Event to a file
    def write(self, common_filename=None, shower_filename=None, efields_filename=None, voltages_filename=None, run_filename=None, overwrite=False):

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
        for el in globals()["grand_tree_list"]:
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
        self.trun.origin_geoid = self.origin_geoid

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
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.tvoltage.trace = [[np.array(v.trace.x).astype(np.float32) for v in self.voltages], [np.array(v.trace.y).astype(np.float32) for v in self.voltages], [np.array(v.trace.z).astype(np.float32) for v in self.voltages]]
        # self.tvoltage.trace_x = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.voltages]
        # self.tvoltage.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.voltages]

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
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.tefield.trace = [[np.array(v.trace.x).astype(np.float32) for v in self.efields], [np.array(v.trace.y).astype(np.float32) for v in self.efields], [np.array(v.trace.z).astype(np.float32) for v in self.efields]]
        # self.tefield.trace_x = [np.array(v.trace.x).astype(np.float32) for v in self.efields]
        # self.tefield.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.efields]
        # self.tefield.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.efields]
        # self.tefield.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.efields]
        # self.tefield.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.efields]
        # self.tefield.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.efields]

        self.tefield.fill()

    ## Fill the shower tree from this Event
    def fill_shower_tree(self, overwrite=False, filename=None, tree_name="tshower"):
        # Fill only if the tree not initialised yet
        if self.tshower is not None and not overwrite:
            raise TreeExists("The tshower TTree already exists!")

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
        self.tshower.xmax_pos = self.shower.Xmaxpos
        ## Shower azimuth
        self.tshower.azimuth = self.shower.azimuth
        ## Shower zenith
        self.tshower.zenith = self.shower.zenith
        ## Poistion of the core on the ground in the site's reference frame
        self.tshower.shower_core_pos = self.shower.core_ground_pos

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

class EventList:
    """A class giving access/iteration over multiple events"""

    ## The instance of the file with TTrees containing the event. ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    file: ROOT.TFile = None

    def __init__(self, file_name, **kwargs):
        # If TFile was given
        if isinstance(file_name, ROOT.TFile):
            self.file_name = file_name.GetName()
            self.file = file_name
        # String with file name was given
        elif isinstance(file_name, str):
            self.file = ROOT.TFile(file_name, "read")

        # The arguments to be passed to Event.fill_event_from_trees()
        self.init_kwargs = kwargs

    def get_event(self, event_number=None, run_number=None, entry_number=None, fill_event=True, **kwargs):
        """Get specified event from the event list"""

        # Don't allow specifying entry and event/run at the same time, because... what to chose?
        if entry_number is not None and (run_number is not None or event_number is not None):
            print("Please provide only entry_number or event/run_number!")
            return None

        e = Event()
        e.file = self.file

        # If entry/event/run number not specified, take the first entry
        run_entry_number = None
        if entry_number is None and run_number is None and event_number is None:
            entry_number = 0

        if entry_number is not None:
            e._entry_number = entry_number
        else:
            if run_number is None:
                run_number = 0
            if event_number is not None:
                e.run_number=run_number
                e.event_number=event_number
            else:
                print("Please provide event_number and run_number, or entry_number")
                return None

        # Fill the event
        if fill_event:
            # Overwrite the init kwargs with kwargs given here
            if len(kwargs)>0:
                e.fill_event_from_trees(**kwargs)
            else:
                e.fill_event_from_trees(**self.init_kwargs)

        return e

    def get_number_of_events(self):
        """Get the number of events in the list"""

        # ToDo: at the moment assumes the same number of events in all the trees
        df = DataFile(self.file)

        # First, try to get the number of events from tshower
        if "TShower" in df.tree_types:
            return df.tshower.get_entries()
        elif "TEfield" in df.tree_types:
            return df.tefield.get_entries()
        elif "TVoltage" in df.tree_types:
            return df.tvoltage.get_entries()
        elif "TRawVoltage" in df.tree_types:
            return df.trawvoltage.get_entries()
        else:
            print("Can not find any tree to provide the number of events in the file.")
            return None

    ## Return the iterable over self
    def __iter__(self):
        # Always start the iteration with the first entry
        current_entry = 0

        entries_cnt = self.get_number_of_events()

        while current_entry < entries_cnt:
            yield self.get_event(entry_number=current_entry)
            current_entry += 1


## Exception risen if the TTree already exists
class TreeExists(Exception):
    pass

