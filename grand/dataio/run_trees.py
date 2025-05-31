# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field

import numpy as np

from grand.dataio import DataTree, TTreeScalarDesc, NotUniqueEvent, logger, StdStringDesc, TTreeArrayDesc, StdVectorListDesc


@dataclass
## A mother class for classes with Run values
class MotherRunTree(DataTree):
    """A mother class for classes with Run values"""

    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"A run with run_number={self.run_number} already exists in the TTree."
            )

        # Repoen the file in write mode, if it exists
        # Reopening in case of different mode takes here ~0.06 s, in case of the same mode, 0.0005 s, so negligible
        if self._file is not None:
            self._file.ReOpen("update")

        # Fill the tree
        self._tree.Fill()

        # Add the current run_number and event_number to the entry_list
        self._entry_list.append(self.run_number)

    def add_proper_friends(self):
        """Add proper friends to this tree"""
        # Create the indices
        self._tree.BuildIndex("run_number")

    ## List runs in the tree
    def print_list_of_runs(self):
        """List runs in the tree"""
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        print("List of runs in the tree:")
        for i in range(count):
            print(int(runs[i]))

    ## Gets list of runs in the tree together
    def get_list_of_runs(self):
        """Gets list of runs in the tree together"""
        count = self._tree.Draw("run_number", "", "goff")
        runs = self._tree.GetV1()
        return [int(runs[i]) for i in range(count)]

    # Readout the TTree entry corresponding to the run
    def get_run(self, run_no):
        """Readout the TTree entry corresponding to the run"""
        # Make sure we have an int
        run_no = int(run_no)
        # Try to get the run from the tree
        res = self._tree.GetEntryWithIndex(int(run_no))
        # If no such entry, return
        if res == 0 or res == -1:
            logger.error(f"No run with run number {run_no}. Please provide a proper number.")
            return 0

        self.assign_branches()

        return res

    def build_index(self, run_id):
        """Build the tree index (necessary for working with friends)"""
        self._tree.BuildIndex(run_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self):
        """Fills the entry list from the tree"""
        # Fill the entry list if there are some entries in the tree
        if (count := self._tree.Draw("run_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(self._tree.GetV1(), dtype=np.float64, count=count))
            self._entry_list = [int(el) for el in v1]

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree"""
        # If the entry list does not exist, the event is unique
        if self._entry_list and self.run_number in self._entry_list:
            return False

        return True


@dataclass
## A class wrapping around a TTree holding values common for the whole run
class TRun(MotherRunTree):
    """A class wrapping around a TTree holding values common for the whole run"""

    _type: str = "run"

    _tree_name: str = "trun"

    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    run_mode: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment"""
    ## Run's first event
    first_event: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run's first event"""
    ## First event time
    first_event_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First event time"""
    ## Run's last event
    last_event: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run's last event"""
    ## Last event time
    last_event_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Last event time"""

    # These are not from the hardware
    ## Data source: detector, sim, other
    data_source: StdStringDesc = field(default=StdStringDesc("detector"))
    """Data source: detector, sim, other"""
    ## Data generator: gtot (in this case)
    data_generator: StdStringDesc = field(default=StdStringDesc("GRANDlib"))
    """Data generator: gtot (in this case)"""
    ## Generator version: gtot version (in this case)
    data_generator_version: StdStringDesc = field(default=StdStringDesc("0.1.0"))
    """Generator version: gtot version (in this case)"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Event format version of the DAQ
    event_version: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event format version of the DAQ"""
    ## Site name
    # _site: StdVectorList("string") = StdVectorList("string")
    site: StdStringDesc = field(default=StdStringDesc())
    """Site name"""
    ## Site layout
    site_layout: StdStringDesc = field(default=StdStringDesc())
    """Site layout"""
    ## Origin of the coordinate system used for the array
    origin_geoid: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Origin of the coordinate system used for the array"""

    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("int", "unsigned int"))
    """Detector unit (antenna) ID"""
    ## Detector unit (antenna) (lat,lon,alt) position
    du_geoid: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) (lat,lon,alt) position"""
    ## Detector unit (antenna) (x,y,z) position in site's referential
    du_xyz: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) (x,y,z) position in site's referential"""
    ## Detector unit type
    du_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))
    """Detector unit type"""
    ## Detector unit (antenna) angular tilt
    du_tilt: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Detector unit (antenna) angular tilt"""
    ## Angular tilt of the ground at the antenna
    du_ground_tilt: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Angular tilt of the ground at the antenna"""
    ## Detector unit (antenna) nut ID
    du_nut: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """Detector unit (antenna) nut ID"""
    ## Detector unit (antenna) FrontEnd Board ID
    du_feb: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """Detector unit (antenna) FrontEnd Board ID"""
    ## Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)
    t_bin_size: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Time bin size in ns (for hardware, computed as 1/adc_sampling_frequency)"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## General info on the voltage common to all events.
class TRunVoltage(MotherRunTree):
    """General info on the voltage common to all events."""

    _type: str = "runvoltage"

    _tree_name: str = "trunvoltage"

    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual).
    digi_ctrl: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual)."""
    ## Firmware version
    firmware_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Firmware version"""
    ## Nominal trace length in units of samples
    trace_length: StdVectorListDesc = field(default=StdVectorListDesc("vector<int>"))
    """Nominal trace length in units of samples"""
    ## ADC sampling frequency in MHz
    adc_sampling_frequency: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling frequency in MHz"""
    ## ADC sampling resolution in bits
    adc_sampling_resolution: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling resolution in bits"""
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    adc_input_channels: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    adc_enabled_channels: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
    ## Value of the Variable gain amplification on the board
    gain: StdVectorListDesc = field(default=StdVectorListDesc("vector<int>"))
    """Value of the Variable gain amplification on the board"""
    ## Conversion factor from bits to V for ADC
    adc_conversion: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Conversion factor from bits to V for ADC"""
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    digi_prepost_trig_windows: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
    ## Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_x: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel x properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_y: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel y properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?
    channel_properties_z: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel z properties - described in Channel property parameters in the manual. ToDo: Decode?"""
    ## Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_x: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel x trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    ## Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_y: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel y trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
    ## Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    channel_trig_settings_z: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel z trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## General info on the raw voltage common to all events.
class TRunRawVoltage(MotherRunTree):
    """General info on the voltage common to all events."""

    _type: str = "runrawvoltage"

    _tree_name: str = "trunrawvoltage"

    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
    ## Firmware version
    firmware_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Firmware version"""
    ## ADC sampling frequency in MHz
    adc_sampling_frequency: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling frequency in MHz"""
    ## ADC sampling resolution in bits
    adc_sampling_resolution: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling resolution in bits"""

    ## ADC input channels for fv2: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4, 4:FILTERED ADC1, 5:FILTERED ADC2, 6:FILTERED ADC3, 7:FILTERED ADC4, 15: off
    adc_input_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """ADC input channels for fv2: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4, 4:FILTERED ADC1, 5:FILTERED ADC2, 6:FILTERED ADC3, 7:FILTERED ADC4, 15: off"""

    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT
    adc_enabled_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT"""

    ## Digitizer window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). The unit is samples (to get time, multiply by 2).
    pre_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Digitizer window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). The unit is samples (to get time, multiply by 2)."""
    ## Digitizer window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). The unit is samples (to get time, multiply by 2).
    post_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Digitizer window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). The unit is samples (to get time, multiply by 2)."""

    ## Channel property parameters
    gain_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel property parameters"""
    integration_time_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    offset_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    base_maximum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    base_minimum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    ## Channel trigger parameters
    signal_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Channel trigger parameters"""
    noise_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    ## The unit is samples (to get time, multiply by 2)
    tper_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """The unit is samples (to get time, multiply by 2)"""
    tprev_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    ncmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    tcmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    qmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    ncmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    qmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    notch_filters_no_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))

    ## Trace length
    trace_length: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Trace length"""

    ## ADC to voltage conversion factor
    adc_conversion: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """ADC to voltage conversion factor"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## The class for storing Efield sim-only data common for a whole run
class TRunEfieldSim(MotherRunTree):
    """The class for storing Efield sim-only data common for a whole run"""

    _type: str = "runefieldsim"

    _tree_name: str = "trunefieldsim"

    ## Name of the atmospheric index of refraction model
    refractivity_model: StdStringDesc = field(default=StdStringDesc())
    """Name of the atmospheric index of refraction model"""
    refractivity_model_parameters: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    ## Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)
    t_pre: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Starting time of antenna data collection time window (because it can be a shorter trace then voltage trace, and thus these parameters can be different)"""
    ## Finishing time of antenna data collection time window
    t_post: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Finishing time of antenna data collection time window"""

    ## Site for which the efield simulation was done
    site: StdStringDesc = field(default=StdStringDesc())
    """Site for which the efield simulation was done"""
    ## Efield simulator name (ZHAireS, Corsika, etc.)
    sim_name: StdStringDesc = field(default=StdStringDesc())
    """Simulator name (aires/corsika, etc.)"""
    ## Efield simulator version string
    sim_version: StdStringDesc = field(default=StdStringDesc())
    """Simulator version string"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## The class for storing shower sim-only data common for a whole run
class TRunShowerSim(MotherRunTree):
    """Run-level info associated with simulated showers"""

    _type: str = "runshowersim"

    _tree_name: str = "trunshowersim"

    ## relative thinning energy
    rel_thin: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """relative thinning energy"""
    # maximum_weight (weight factor)
    maximum_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax"""

    hadronic_thinning: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of energy at wich thining starts in hadrons and electromagnetic particles"""
    hadronic_thinning_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """the ratio of electromagnetic to hadronic maximum weights"""

    ## low energy cut for electrons (GeV)
    lowe_cut_e: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for electrons (GeV)"""
    ## low energy cut for gammas (GeV)
    lowe_cut_gamma: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for gammas (GeV)"""
    ## low energy cut for muons (GeV)
    lowe_cut_mu: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for muons (GeV)"""
    ## low energy cut for mesons (GeV)
    lowe_cut_meson: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for mesons (GeV)"""
    ## low energy cut for nuceleons (GeV)
    lowe_cut_nucleon: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """low energy cut for nucleons (GeV)"""
    ## Site for which the shower simulation was done
    site: StdStringDesc = field(default=StdStringDesc())
    """Site for which the shower simulation was done"""
    ## Simulator name (aires/corsika, etc.)
    sim_name: StdStringDesc = field(default=StdStringDesc())
    """Simulator name (aires/corsika, etc.)"""
    ## Simulator version string
    sim_version: StdStringDesc = field(default=StdStringDesc())
    """Simulator version string"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()


@dataclass
## General info on the noise generation
class TRunNoise(MotherRunTree):
    """General info on the noise generation"""

    _type: str = "runnoise"

    _tree_name: str = "trunnoise"

    ## Info to retrieve the map of galactic noise
    gal_noise_map: StdStringDesc = field(default=StdStringDesc())
    """Info to retrieve the map of galactic noise"""
    ## LST time when we generate the noise
    gal_noise_LST: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """LST time when we generate the noise"""
    ## Noise std dev for each arm of each antenna
    gal_noise_sigma: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Noise std dev for each arm of each antenna"""

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()
