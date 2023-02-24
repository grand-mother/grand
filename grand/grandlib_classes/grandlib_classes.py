## The grandlib classes following https://docs.google.com/document/d/1P0AwR3U3MVZyU1ewIobWkJPZmVkxKCAw/edit

from dataclasses import dataclass, field, fields
import numpy as np
from typing import Any
from grand.io.root_trees import *
from grand.tools.coordinates import *
import ROOT

## A class describing a single antenna; ToDo: Should it be antenna, or more general: Detector?
@dataclass
class Antenna():
    ## Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    ## Antenna tilt
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    ## Antenna acceleration - this comes from hardware. ToDo: perhaps recalculate to tilt or remove tilt?
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    ## The antenna model
    model: Any = 0

    # ToDo: Parameters below come from the hardware, but do we want them here?
    ## Atmospheric temperature (read via I2C)
    atm_temperature: float = 0
    ## Atmospheric pressure
    atm_pressure: float = 0
    ## Atmospheric humidity
    atm_humidity: float = 0
    ## Battery voltage
    battery_level: float = 0
    ## Firmware version
    firmware_version: float = 0

    ## Maybe these should go somehow to the Timetrace3D?
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

    # I don't think it is useful for the user
    # ## Clock tick at which the event was triggered (used to calculate the trigger time)
    # clock_tick: np.uint16 = 0
    # ## Clock ticks per second
    # clock_ticks_per_second: np.uint16 = 0
    # ## GPS offset - offset between the PPS and the real second (in GPS).
    # gps_offset: float = 0
    # ## GPS leap second
    # gps_leap_second: np.uint16 = 0
    # ## GPS status
    # gps_status: np.uint16 = 0
    # ## GPS alarms
    # gps_alarms: np.uint16 = 0
    # ## GPS warnings
    # gps_warnings: np.uint16 = 0
    # ## GPS time
    # gps_time: int = 0
    # ## GPS temperature
    # gps_temp: float = 0


## A class for holding x,y,z single antenna traces over time
@dataclass
class Timetrace3D:
    ## The trace length
    n_points: int = 0
    ## [ns] n_points x step = total timetrace length
    time_step: float = 0
    ## Start time as unix time with nanoseconds
    t0: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))
    ## Trigger time as unix time with nanoseconds
    trigger_time: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))

    ## Trace vector in X
    # trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Y
    # trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Z
    # trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    _trace: CartesianRepresentation = field(default_factory=lambda: np.zeros(1, np.float))
    # _trace1: list = None
    # trace: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
    ## Hilbert envelope vector in X
    hilbert_trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Hilbert envelope vector in X
    hilbert_trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Hilbert envelope vector in X
    hilbert_trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    ## ToDo: add additional quantities from the doc?

    ## ToDo: add additional quantities from the trees?

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, v):
        self._trace = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

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
    eta: float = 0
    ## Ratio of the geomagnetic to charge excess contributions
    a_ratio: float = 0

## A class for holding a shower
@dataclass
class Shower():
    ## Shower energy [eV]
    energy: float = 0
    ## Shower Xmax [g/cm2]
    Xmax: float = 0
    ## Shower position in the site's reference frame
    Xmaxpos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
    origin_geoid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ## Poistion of the core on the ground in the site's reference frame
    core_ground_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

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
    ## Was this event reconstructed?
    is_reconstructed: bool = False
    ## Is this event associated to a single wave based on reconstruction
    is_wave: bool = False
    ## Vector of origin of plane wave fit
    origin_planewave: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    ## Chi2 of plane wave fit
    chi2_planewave: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    ## Position of the source according to spherical fit
    origin_sphere: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    ## Chi2 of spherical fit
    chi2_sphere: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    ## Is this an EAS?
    is_eas: bool = False

    ## Reconstructed shower
    shower: Shower() = None

    ## Simualted shower for simulations
    simShower: Shower() = None

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
    origin_geoid: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    # Internal trees
    trun: ROOT.TTree = None
    teventvoltage: ROOT.TTree = None
    teventefield: ROOT.TTree = None
    teventshower: ROOT.TTree = None

    # Options

    # Close files automatically after event write? - slower writing but less maitanance by the user
    auto_file_close: bool = True

    ## Post-init actions, like an automatic readout from files, etc.
    def __post_init__(self):
        # If the file name and event number was given, init the Event from trees
        if self.file and self.event_number:
            self.fill_event_from_trees()

    ## Fill this event from trees
    def fill_event_from_trees(self):
        # Check if the file exist
        if not self.file:
            print("No file provided to init from. Aborting.")
            return False

        # If the _file is not yet TFile, make it so
        if not isinstance(self.file, ROOT.TFile):
            self.file = ROOT.TFile(self.file, "read")

        # *** Check what TTrees are available and fill according to their availability

        # Check the Run tree existence
        if trun := self.file.Get("trun"):
            self.trun = RunTree(_tree=trun)
            # Fill part of the event from trun
            self.fill_event_from_runtree()
        else:
            print("No Run tree. Run information will not be available.")
            # Make trun really None
            self.trun = None

        # Check the EventVoltage tree existence
        if teventvoltage := self.file.Get("teventvoltage"):
            self.teventvoltage = VoltageEventTree(_tree=teventvoltage)
            # Fill part of the event from teventvoltage
            self.fill_event_from_eventvoltage_tree()
        else:
            print("No EventVoltage tree. Voltage information will not be available.")
            # Make teventvoltage really None
            self.teventvoltage = None

        # Check the EventEfield tree existence
        if teventefield := self.file.Get("teventefield"):
            self.teventefield = EfieldEventTree(_tree=teventefield)
            # Fill part of the event from teventefield
            self.fill_event_from_eventefield_tree()
        else:
            print("No Eventefield tree. Efield information will not be available.")
            # Make teventefield really None
            self.teventefield = None

        # Check the EventShower tree existence
        if teventshower := self.file.Get("teventshower"):
            self.teventshower = ShowerEventTree(_tree=teventshower)
            # Fill part of the event from teventshower
            self.fill_event_from_eventshower_tree()
        else:
            print("No EventShower tree. Shower information will not be available.")
            # Make teventshower really None
            self.teventshower = None


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
        self.antennas = []
        # Loop through traces
        for i in range(len(self.teventvoltage.trace_x)):
            # Fill the voltage trace part
            v = Voltage()
            tx = self.teventvoltage.trace_x[i]
            v.n_points = len(tx)
            v.trace_x = tx
            v.trace_y = self.teventvoltage.trace_y[i]
            v.trace_z = self.teventvoltage.trace_z[i]
            self.voltages.append(v)

            # Fill the antenna part
            a = Antenna()
            a.atm_temperature = self.teventvoltage.atm_temperature[i]
            a.atm_pressure = self.teventvoltage.atm_pressure[i]
            a.atm_humidity = self.teventvoltage.atm_humidity[i]
            a.battery_level = self.teventvoltage.battery_level[i]
            a.firmware_version = self.teventvoltage.firmware_version[i]
            self.antennas.append(a)

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

    ## Fill part of the event from the EventEfield tree
    def fill_event_from_eventefield_tree(self):
        # Initialise the Shower
        self.shower = Shower()
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

    ## Fill part of the event from the EventShower tree
    def fill_event_from_eventshower_tree(self):
        self.teventshower.get_event(self.event_number, self.run_number)
        ## Shower energy [eV]
        self.shower.energy = self.teventshower.shower_energy
        ## Shower Xmax [g/cm2]
        self.shower.Xmax = self.teventshower.xmax_grams
        ## Shower position in the site's reference frame
        self.shower.Xmaxpos = self.teventshower.xmax_pos_shc
        ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
        self.shower.origin_geoid = field(default_factory=lambda: np.zeros(3))
        ## Poistion of the core on the ground in the site's reference frame
        self.shower.core_ground_pos = self.teventshower.shower_core_pos

    ## Print all the class values
    def print(self):
        # Assign the TTree branches to the class fields
        for field in fields(self):
            # Skip the list fields
            if any(x in field.name for x in {"antennas", "voltages", "efields", "shower"}): continue
            print("{:<30} {:>30}".format(field.name, str(getattr(self, field.name))))

        # Now deal with the list fields separately

        print("Shower:")
        print("\t{:<30} {:>30}".format("Energy:", self.shower.energy))
        print("\t{:<30} {:>30}".format("Xmax [g/cm2]:", self.shower.Xmax))
        print("\t{:<30} {:>30}".format("Xmax position:", str(self.shower.Xmaxpos)))
        print("\t{:<30} {:>30}".format("Origin geoid:", str(self.shower.origin_geoid)))
        print("\t{:<30} {:>30}".format("Core ground pos:", str(self.shower.core_ground_pos)))

        print("Antennas:")
        print("\t{:<30} {:>30}".format("No of antennas:", len(self.antennas)))
        print("\t{:<30} {:>30}".format("Position:", str([a.position for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Tilt:", str([a.tilt for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Acceleration:", str([a.acceleration for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Humidity:", str([a.atm_humidity for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Pressure:", str([a.atm_pressure for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Temperature:", str([a.atm_temperature for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Battery level:", str([a.battery_level for a in self.antennas])))
        print("\t{:<30} {:>30}".format("Firmware version:", str([a.firmware_version for a in self.antennas])))

        print("Voltages:")
        print("\t{:<30} {:>30}".format("Triggered status:", str([tr.is_triggered for tr in self.voltages])))
        print("\t{:<30} {:>30}".format("Traces lengths:", str([len(tr.trace_x) for tr in self.voltages])))
        print("\t{:<30} {:>30}".format("Traces first values:", str([tr.trace_x[0] for tr in self.voltages])))

        print("Efields:")
        print("\t{:<30} {:>30}".format("Traces lengths:", str([len(tr.trace_x) for tr in self.efields])))
        print("\t{:<30} {:>30}".format("Traces first values:", str([tr.trace_x[0] for tr in self.efields])))

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
            self.teventvoltage.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)

    ## Write the efields to a file
    def write_efields(self, filename, overwrite=False):
        self.fill_efield_tree(filename=filename)
        if self.auto_file_close:
            self.teventefield.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)

    ## Write the shower to a file
    def write_shower(self, filename, overwrite=False):
        self.fill_shower_tree(filename=filename)
        if self.auto_file_close:
            self.teventshower.write(filename, overwrite=overwrite, force_close_file=self.auto_file_close)


    ## Fill the run tree from this Event
    def fill_run_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.trun is not None and not overwrite:
            raise TreeExists("The trun TTree already exists!")

        # Look for the RunTree with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the RunTree with the same file and name in the memory exists, use it
            if type(el)==RunTree and el._tree_name=="trun" and el._file_name==filename:
                self.trun = el
                break
        # No same RunTree in memory - create a new one
        else:
            self.trun = RunTree(_file_name=filename, _tree_name="trun")

        # Copy the event into the tree
        self.trun.run_number = self.run_number
        self.trun.run_mode = self.run_mode
        self.trun.data_source = self.data_source
        self.trun.data_generator = self.data_generator
        self.trun.data_generator_version = self.data_generator_version
        self.trun.site = self.site
        self.trun.site_long = self.site_long
        self.trun.site_lat = self.site_lat
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
        if self.teventvoltage is not None and not overwrite:
            raise TreeExists("The teventvoltage TTree already exists!")

        # Look for the VoltageEventTree with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the VoltageEventTree with the same file and name in the memory exists, use it
            if type(el)==VoltageEventTree and el._tree_name=="teventvoltage" and el._file_name==filename:
                self.teventvoltage = el
                break
        # No same VoltageEventTree in memory - create a new one
        else:
            self.teventvoltage = VoltageEventTree(_file_name = filename)

        self.teventvoltage.run_number = self.run_number
        self.teventvoltage.event_number = self.event_number

        # Copy the contents of voltages to the tree
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.teventvoltage.trace_x = [np.array(v.trace.x).astype(np.float32) for v in self.voltages]
        self.teventvoltage.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.voltages]
        self.teventvoltage.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.voltages]
        # self.teventvoltage.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.voltages]
        # self.teventvoltage.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.voltages]
        # self.teventvoltage.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.voltages]

        # Copy the contents of antennas to the tree
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.teventvoltage.atm_temperature = np.array([np.array(a.atm_temperature) for a in self.antennas])
        self.teventvoltage.atm_pressure = np.array([np.array(a.atm_pressure) for a in self.antennas])
        self.teventvoltage.atm_humidity = np.array([np.array(a.atm_humidity) for a in self.antennas])
        self.teventvoltage.battery_level = np.array([np.array(a.battery_level) for a in self.antennas])
        self.teventvoltage.firmware_version = np.array([np.array(a.firmware_version) for a in self.antennas])

        self.teventvoltage.fill()

    ## Fill the efield tree from this Event
    def fill_efield_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.teventefield is not None and not overwrite:
            raise TreeExists("The teventefield TTree already exists!")

        # Look for the EfieldEventTree with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the EfieldEventTree with the same file and name in the memory exists, use it
            if type(el)==EfieldEventTree and el._tree_name=="teventefield" and el._file_name==filename:
                self.teventefield = el
                break
        # No same EfieldEventTree in memory - create a new one
        else:
            self.teventefield = EfieldEventTree(_file_name = filename)

        self.teventefield.run_number = self.run_number
        self.teventefield.event_number = self.event_number

        # Copy the contents of efields to the tree
        # Remark: best to set list. Append will append to the previous event, since it is not cleared automatically
        self.teventefield.trace_x = [np.array(v.trace.x).astype(np.float32) for v in self.efields]
        self.teventefield.trace_y = [np.array(v.trace.y).astype(np.float32) for v in self.efields]
        self.teventefield.trace_z = [np.array(v.trace.z).astype(np.float32) for v in self.efields]
        # self.teventefield.trace_x = [np.array(v.trace_x).astype(np.float32) for v in self.efields]
        # self.teventefield.trace_y = [np.array(v.trace_y).astype(np.float32) for v in self.efields]
        # self.teventefield.trace_z = [np.array(v.trace_z).astype(np.float32) for v in self.efields]

        self.teventefield.fill()

    ## Fill the shower tree from this Event
    def fill_shower_tree(self, overwrite=False, filename=None):
        # Fill only if the tree not initialised yet
        if self.teventshower is not None and not overwrite:
            raise TreeExists("The teventshower TTree already exists!")

        # Look for the ShowerEventTree with the same file and name in the memory
        for el in globals()["grand_tree_list"]:
            # If the ShowerEventTree with the same file and name in the memory exists, use it
            if type(el)==ShowerEventTree and el._tree_name=="teventshower" and el._file_name==filename:
                self.teventshower = el
                break
        # No same ShowerEventTree in memory - create a new one
        else:
            self.teventshower = ShowerEventTree(_file_name = filename)

        self.teventshower.run_number = self.run_number
        self.teventshower.event_number = self.event_number


        self.teventshower.shower_energy = self.shower.energy
        ## Shower Xmax [g/cm2]
        self.teventshower.xmax_grams = self.shower.Xmax
        ## Shower position in the site's reference frame
        self.teventshower.xmax_pos_shc = self.shower.Xmaxpos
        ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
        #self.shower.origin_geoid = np.zeros(3)
        ## Poistion of the core on the ground in the site's reference frame
        self.teventshower.shower_core_pos = self.shower.core_ground_pos

        self.teventshower.fill()

    def close_files(self):
        """Close all files of the all trees - needed when auto_file_close is False"""
        self.teventshower.write()
        self.teventefield.write()
        self.teventvoltage.write()
        self.trun.write()
        self.teventshower.close_file()
        self.teventefield.close_file()
        self.teventvoltage.close_file()
        self.trun.close_file()



## Exception risen if the TTree already exists
class TreeExists(Exception):
    pass

