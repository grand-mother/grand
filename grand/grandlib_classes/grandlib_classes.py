## The grandlib classes following https://docs.google.com/document/d/1P0AwR3U3MVZyU1ewIobWkJPZmVkxKCAw/edit

from dataclasses import dataclass, field
import numpy as np
from typing import Any
from grand.io.root_trees import *

## A class describing a single antenna
@dataclass
class Antenna():
    ## Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)
    _position : np.ndarray = np.zeros(3, np.float32)
    ## Antenna tilt
    _tilt : np.ndarray = np.zeros(3, np.float32)
    ## Antenna acceleration - this comes from hardware. ToDo: perhaps recalculate to tilt or remove tilt?
    _acceleration : np.ndarray = np.zeros(3, np.float32)

    ## The antenna model
    _model : Any = 0

    # Parameters below come from the hardware, but do we want them here?
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: float = 0
    ## Atmospheric pressure
    _atm_pressure: float = 0
    ## Atmospheric humidity
    _atm_humidity: float = 0
    ## Battery voltage
    _battery_level: float = 0
    ## Firmware version
    _firmware_version: float = 0

    ## Maybe these should go some how to the Timetrace3D?
    # ## ADC sampling frequency in MHz
    # _adc_sampling_frequency: float = 0
    # ## ADC sampling resolution in bits
    # _adc_sampling_resolution: float = 0
    # ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    # _adc_input_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    # _adc_enabled_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC samples callected in all channels
    # _adc_samples_count_total: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC samples callected in channel 0
    # _adc_samples_count_channel0: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC samples callected in channel 1
    # _adc_samples_count_channel1: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC samples callected in channel 2
    # _adc_samples_count_channel2: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## ADC samples callected in channel 3
    # _adc_samples_count_channel3: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    # _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # ## Trigger rate - the number of triggers recorded in the second preceding the event
    # _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")

    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: np.uint16 = 0
    ## Clock ticks per second
    _clock_ticks_per_second: np.uint16 = 0
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: float = 0
    ## GPS leap second
    _gps_leap_second: np.uint16 = 0
    ## GPS status
    _gps_status: np.uint16 = 0
    ## GPS alarms
    _gps_alarms: np.uint16 = 0
    ## GPS warnings
    _gps_warnings: np.uint16 = 0
    ## GPS time
    _gps_time: int = 0
    ## GPS temperature
    _gps_temp: float = 0



## A class for holding x,y,z single antenna traces over time
@dataclass
class Timetrace3D():
    ## The trace length
    n_points: int = 0
    ## [ns] n_points x step = total timetrace length
    time_step: float = 0
    ## Start time as unix time with nanoseconds
    t0: np.datetime64 = np.datetime64(0, 'ns')
    ## Trigger time as unix time with nanoseconds
    trigger_time: np.datetime64 = np.datetime64(0, 'ns')

    ## Trace vector in X
    trace_x: np.ndarray = np.zeros(1, np.float)
    ## Trace vector in Y
    trace_y: np.ndarray = np.zeros(1, np.float)
    ## Trace vector in Z
    trace_z: np.ndarray = np.zeros(1, np.float)

    ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
    ## Hilbert envelope vector in X
    hilbert_trace_x: np.ndarray = np.zeros(1, np.float)
    ## Hilbert envelope vector in X
    hilbert_trace_y: np.ndarray = np.zeros(1, np.float)
    ## Hilbert envelope vector in X
    hilbert_trace_z: np.ndarray = np.zeros(1, np.float)

    ## ToDo: add additional quantities from the doc?

    ## ToDo: add additional quantities from the trees?

## A class for holding voltage traces + additional information
@dataclass
class Voltage(Timetrace3D):
    ## GPS time of the trigger - why would we want it? We have already _trigger_time in Timetrace3D, that is GPS time + nanoseconds
    # _GPS_trigtime: np.uint32 = 0
    ## Is this a triggered trace? - not sure if it should be here or in Timetrace3D, or perhaps further up in the event
    is_triggered: bool = True

## A class for holding Efield traces + additional information
@dataclass
class Efield(Timetrace3D):
    ## Polarization angle of the reconstructed Efield in the shower plane [deg]
    _eta: float = 0
    ## Ratio of the geomagnetic to charge excess contributions
    _a_ratio: float = 0

## A class for holding a shower
@dataclass
class Shower():
    ## Shower energy [eV]
    _energy: float = 0
    ## Shower Xmax [g/cm2]
    _Xmax: float = 0
    ## Shower position in the site's reference frame
    _Xmaxpos: np.ndarray = np.zeros(3, dtype=np.float32)
    ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
    _origin_geoid: np.ndarray = np.zeros(3, dtype=np.float32)
    ## Poistion of the core on the ground in the site's reference frame
    _core_ground_pos: np.ndarray = np.zeros(3, dtype=np.float32)

## A class for holding an event
@dataclass
class Event():
    ## The instance of the file with TTrees containing the event. ToDo: this should allow for multiple files holding different TTrees and TChains in the future
    file: ROOT.TFile = None

    ## The current event in the file number
    event_number: int = None
    ## The run number of the current event
    run_number: int = None


    ## Antennas participating in the event
    antennas: list[Antenna] = None
    ## Event multiplicity: ToDo: what is it?
    L: int = 0
    ## Voltages from different antennas
    voltages: list[Voltage] = None
    ## Efields from different antennas
    efields: list[Efield] = None

    # Reconstruction parameters
    ## Is this event associated to a single wave based on reconstruction
    _is_wave: bool = False
    ## Vector of origin of plane wave fit
    _origin_planewave: np.ndarray = np.zeros(3, np.float32)
    ## Chi2 of plane wave fit
    _chi2_planewave: np.ndarray = np.zeros(3, np.float32)
    ## Position of the source according to spherical fit
    _origin_sphere: np.ndarray = np.zeros(3, np.float32)
    ## Chi2 of spherical fit
    _chi2_sphere: np.ndarray = np.zeros(3, np.float32)

    ## Is this an EAS?
    _is_eas: bool = False

    ## Reconstructed shower
    _shower: Shower() = None

    ## Simualted shower for simulations
    _simShower: Shower() = None

    # *** Run related properties
    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    run_mode: np.uint32 = 0
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
    ## Data source: detector, simulation, other
    data_source: str = "other"
    ## Data generator: gtot (in this case)
    data_generator: str = "GRANDlib"
    ## Generator version: gtot version (in this case)
    data_generator_version: str = "0.1.0"
    ## Site name
    site: str = "DummySite"
    ## Site longitude
    site_long: np.float32 = 0
    ## Site latitude
    site_lat: np.float32 = 0
    ## Origin of the coordinate system used for the array
    origin_geoid: np.ndarray = np.zeros(3, np.float32)


    ## Post-init actions, like an automatic readout from files, etc.
    def __post_init__(self):
        # If the file name and event number was given, init the Event from trees
        if self.file and self.event_number:
            self.fill_event_from_trees()

    ## Fill this event from trees
    def fill_event_from_trees(self):
        print(self.file)
        # Check if the file exist
        if not self.file:
            print("No file provided to init from. Aborting.")
            return False

        # If the _file is not yet TFile, make it so
        if not isinstance(self.file, ROOT.TFile):
            self.file = ROOT.TFile(self.file, "read")

        # *** Check what TTrees are available and fill according to their availability

        # Check the Run tree existence
        if trun:=self.file.Get("trun"):
            self.trun = RunTree(_tree=trun)
            # Fill part of the event from trun
            self.fill_event_from_runtree()
        else:
            print("No Run tree. Run information will not be available.")
            # Make trun really None
            self.trun = None

        # Check the EventVoltage tree existence
        if teventvoltage:=self.file.Get("teventvoltage"):
            self.teventvoltage = VoltageEventTree(_tree=teventvoltage)
            # Fill part of the event from teventvoltage
            self.fill_event_from_eventvoltage_tree()
        else:
            print("No EventVoltage tree. Voltage information will not be available.")
            # Make teventvoltage really None
            self.teventvoltage = None

        # Check the EventEfield tree existence
        if teventefield:=self.file.Get("teventefield"):
            self.teventefield = EfieldEventTree(_tree=teventefield)
            # Fill part of the event from teventefield
            self.fill_event_from_eventefield_tree()
        else:
            print("No Eventefield tree. Efield information will not be available.")
            # Make teventefield really None
            self.teventefield = None

    ## Fill part of the event from the Run tree
    def fill_event_from_runtree(self):
        # Read the event into the class
        self.trun.get_run(self.run_number)

        # Copy the values
        self.run_mode = self.trun.run_mode
        self.data_source = self.trun.data_source
        self.data_generator = self.trun.data_generator
        self.data_generator_version = self.trun.data_generator_version
        self.site = self.trun.site
        self.site_long = self.trun.site_long
        self.site_lat = self.trun.site_lat
        self.origin_geoid = self.trun.origin_geoid

    ## Fill part of the event from the EventVoltage tree
    def fill_event_from_eventvoltage_tree(self):
        self.teventvoltage.get_event(self.event_number, self.run_number)
        self.voltages = []
        # Loop through traces
        for i in range(len(self.teventvoltage.trace_x)):
            v = Voltage()
            tx = self.teventvoltage.trace_x[i]
            v.n_points = len(tx)
            v.trace_x = tx
            v.trace_y = self.teventvoltage.trace_y[i]
            v.trace_z = self.teventvoltage.trace_z[i]
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
        # ## Trace vector in X
        # _trace_x: np.ndarray = np.zeros(1, np.float)
        # ## Trace vector in Y
        # _trace_y: np.ndarray = np.zeros(1, np.float)
        # ## Trace vector in Z
        # _trace_z: np.ndarray = np.zeros(1, np.float)
        #
        # ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
        # ## Hilbert envelope vector in X
        # _hilbert_trace_x: np.ndarray = np.zeros(1, np.float)
        # ## Hilbert envelope vector in X
        # _hilbert_trace_y: np.ndarray = np.zeros(1, np.float)
        # ## Hilbert envelope vector in X
        # _hilbert_trace_z: np.ndarray = np.zeros(1, np.float)

    ## Fill part of the event from the EventEfield tree
    def fill_event_from_eventefield_tree(self):
        self.teventefield.get_event(self.event_number, self.run_number)
        self.efields = []
        # Loop through traces
        for i in range(len(self.teventefield.trace_x)):
            v = Efield()
            tx = self.teventefield.trace_x[i]
            v.n_points = len(tx)
            v.trace_x = tx
            v.trace_y = self.teventefield.trace_y[i]
            v.trace_z = self.teventefield.trace_z[i]
            self.efields.append(v)


    ## Print all the class values
    def print(self):
        # Assign the TTree branches to the class fields
        for field in self.__dataclass_fields__:
            # Skip "tree" and "file" fields, as they are not the part of the stored data
            # if field == "_tree" or field == "_file" or field == "_tree_name" or field == "_cur_du_id": continue
            print(field, self.__dataclass_fields__[field])
            # Read the TTree branch
            # u = getattr(self._tree, field[1:])
            # print(self.__dataclass_fields__[field].name, u, type(u))
