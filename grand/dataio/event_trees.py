# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field

import numpy as np
import ROOT

from grand.dataio import DataTree, TTreeScalarDesc, NotUniqueEvent, grand_tree_list, TRun, logger, StdVectorListDesc, StdStringDesc, TTreeArrayDesc


@dataclass
## A mother class for classes with Event values
class MotherEventTree(DataTree):
    """A mother class for classes with Event values"""

    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    event_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    # ## Create metadata for the tree
    # def create_metadata(self):
    #     """Create metadata for the tree"""
    #     # First add the medatata of the mother class
    #     super().create_metadata()
    #     # ToDo: stupid, because default values are generated here and in the class fields definitions. But definition of the class field does not call the setter, which is needed to attach these fields to the tree.
    #     self.source_datetime = datetime.datetime.fromtimestamp(0)
    #     self.modification_software = ""
    #     self.modification_software_version = ""
    #     self.analysis_level = 0

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"An event with (run_number,event_number)=({self.run_number},{self.event_number}) already exists in the TTree {self._tree.GetName()}."
            )

        # Repoen the file in write mode, if it exists
        # Reopening in case of different mode takes here ~0.06 s, in case of the same mode, 0.0005 s, so negligible
        if self._file is not None:
            self._file.ReOpen("update")

        # Fill the tree
        self._tree.Fill()

        # If there is no entry list, create it
        if not self._entry_list:
            self.fill_entry_list()
        # Add the current run_number and event_number to the entry_list
        self._entry_list.append((self.run_number, self.event_number))

    def add_proper_friends(self):
        """Add proper friends to this tree"""
        # Create the indices
        self.build_index("run_number", "event_number")
        # For now, do not add friends
        return 0

        # Add the Run tree as a friend if exists already
        loc_vars = dict(locals())
        run_trees = []
        for inst in grand_tree_list:
            if type(inst) is TRun:
                run_trees.append(inst)
        # If any Run tree was found
        if len(run_trees) > 0:
            # Warning if there is more than 1 TRun in memory
            if len(run_trees) > 1:
                logger.warning(
                    f"More than 1 TRun detected in memory. Adding the last one {run_trees[-1]} as a friend"
                )
            # Add the last one TRun as a friend
            run_tree = run_trees[-1]

            # Add the Run TTree as a friend
            self.add_friend(run_tree.tree, run_tree.file)

        # Do not add TADC as a friend to itself
        if not isinstance(self, TADC):
            # Add the ADC tree as a friend if exists already
            adc_trees = []
            for inst in grand_tree_list:
                if type(inst) is TADC:
                    adc_trees.append(inst)
            # If any ADC tree was found
            if len(adc_trees) > 0:
                # Warning if there is more than 1 TADC in memory
                if len(adc_trees) > 1:
                    logger.warning(
                        f"More than 1 TADC detected in memory. Adding the last one {adc_trees[-1]} as a friend"
                    )
                # Add the last one TADC as a friend
                adc_tree = adc_trees[-1]

                # Add the ADC TTree as a friend
                self.add_friend(adc_tree.tree, adc_tree.file)

        # Do not add TRawVoltage as a friend to itself
        if not isinstance(self, TRawVoltage):
            # Add the Voltage tree as a friend if exists already
            voltage_trees = []
            for inst in grand_tree_list:
                if type(inst) is TRawVoltage:
                    voltage_trees.append(inst)
            # If any voltage tree was found
            if len(voltage_trees) > 0:
                # Warning if there is more than 1 TRawVoltage in memory
                if len(voltage_trees) > 1:
                    logger.warning(
                        f"More than 1 TRawVoltage detected in memory. Adding the last one {voltage_trees[-1]} as a friend"
                    )
                # Add the last one TRawVoltage as a friend
                voltage_tree = voltage_trees[-1]

                # Add the Voltage TTree as a friend
                self.add_friend(voltage_tree.tree, voltage_tree.file)

        # Do not add TEfield as a friend to itself
        if not isinstance(self, TEfield):
            # Add the Efield tree as a friend if exists already
            efield_trees = []
            for inst in grand_tree_list:
                if type(inst) is TEfield:
                    efield_trees.append(inst)
            # If any Efield tree was found
            if len(efield_trees) > 0:
                # Warning if there is more than 1 TEfield in memory
                if len(efield_trees) > 1:
                    logger.warning(
                        f"More than 1 TEfield detected in memory. Adding the last one {efield_trees[-1]} as a friend"
                    )
                # Add the last one TEfield as a friend
                efield_tree = efield_trees[-1]

                # Add the Efield TTree as a friend
                self.add_friend(efield_tree.tree, efield_tree.file)

        # Do not add TShower as a friend to itself
        if not isinstance(self, TShower):
            # Add the Shower tree as a friend if exists already
            shower_trees = []
            for inst in grand_tree_list:
                if type(inst) is TShower:
                    shower_trees.append(inst)
            # If any Shower tree was found
            if len(shower_trees) > 0:
                # Warning if there is more than 1 TShower in memory
                if len(shower_trees) > 1:
                    logger.warning(
                        f"More than 1 TShower detected in memory. Adding the last one {shower_trees[-1]} as a friend"
                    )
                # Add the last one TShower as a friend
                shower_tree = shower_trees[-1]

                # Add the Shower TTree as a friend
                self.add_friend(shower_tree.tree, shower_tree.file)

    ## List events in the tree together with runs
    def print_list_of_events(self):
        """List events in the tree together with runs"""
        count = self._tree.Draw("event_number:run_number", "", "goff")
        # Remove the Draw() generated histogram from current file to prevent saving
        if tmph := ROOT.gDirectory.Get("htemp"):
            tmph.SetDirectory(0)
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        print("List of events in the tree:")
        print("event_number run_number")
        for i in range(count):
            print(int(events[i]), int(runs[i]))

    ## Gets list of events in the tree together with runs
    def get_list_of_events(self):
        """Gets list of events in the tree together with runs"""
        count = self._tree.Draw("event_number:run_number", "", "goff")
        # Remove the Draw() generated histogram from current file to prevent saving
        if tmph := ROOT.gDirectory.Get("htemp"):
            tmph.SetDirectory(0)
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        return [(int(events[i]), int(runs[i])) for i in range(count)]

    ## Readout the TTree entry corresponding to the event and run
    def get_event(self, ev_no, run_no=0):
        """Readout the TTree entry corresponding to the event and run"""
        # Try to get the requested entry
        # res = self._tree.GetEntryWithIndex(int(run_no), int(ev_no))
        # The above should work, but there is a bug in ROOT
        res = self._tree.GetEntry(self._tree.GetEntryNumberWithIndex(int(run_no), int(ev_no)))
        # If no such entry, return
        if res == 0 or res == -1:
            logger.error(
                f"No event with event number {ev_no} and run number {run_no} in the {self.tree_name} tree. Please provide proper numbers."
            )
            return 0

        self.assign_branches()

        return res

    ## Check if the TTree has an entry with the given event and run number
    def has_event(self, ev_no, run_no=0):
        """Check if the TTree has an entry with the given event and run number"""
        # Try to get the requested entry
        res = self._tree.GetEntryNumberWithIndex(int(run_no), int(ev_no))
        # If no such entry, return
        if res == -1:
            return False
        else:
            return True

    ## Builds index based on run_id and evt_id for the TTree
    def build_index(self, run_id, evt_id):
        """Builds index based on run_id and evt_id for the TTree"""
        self._tree.BuildIndex(run_id, evt_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self, tree=None):
        """Fills the entry list from the tree"""
        if tree is None:
            tree = self._tree
        # Fill the entry list if there are some entries in the tree
        if (count := tree.Draw("run_number:event_number", "", "goff")) > 0:
            v1 = np.array(np.frombuffer(tree.GetV1(), dtype=np.float64, count=count))
            v2 = np.array(np.frombuffer(tree.GetV2(), dtype=np.float64, count=count))
            self._entry_list = [(int(el[0]), int(el[1])) for el in zip(v1, v2)]
        # Remove the Draw() generated histogram from current file to prevent saving
        if tmph := ROOT.gDirectory.Get("htemp"):
            tmph.SetDirectory(0)

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree"""
        # If there is no entry list, create it
        if not self._entry_list:
            self.fill_entry_list()
        # If the entry list does not exist, the event is unique
        if self._entry_list and (self.run_number, self.event_number) in self._entry_list:
            return False

        return True

    def get_traces_lengths(self):
        """Gets the traces lengths for each event"""

        # If there are no traces in the tree, return None
        if self._tree.GetListOfLeaves().FindObject("trace_x") == None and self._tree.GetListOfLeaves().FindObject("trace_0") == None:
            return None

        traces_lengths = []
        # For ADC traces - 4 traces, different names
        if "ADC" in self.__class__.__name__ or "RawVoltage" in self.__class__.__name__:
            traces_suffixes = [0, 1, 2, 3]
        # Other traces
        else:
            traces_suffixes = ["x", "y", "z"]

        # Get sizes of each traces
        for i in traces_suffixes:
            cnt = self._tree.Draw(f"@trace_{i}.size()", "", "goff")
            traces_lengths.append(np.frombuffer(self._tree.GetV1(), count=cnt, dtype=np.float64).astype(int).tolist())
            # Remove the Draw() generated histogram from current file to prevent saving
            if tmph := ROOT.gDirectory.Get("htemp"):
                tmph.SetDirectory(0)

        return traces_lengths

    def get_list_of_dus(self):
        """Gets the list of all detector units used for each event"""

        # If there are no detector unit ids in the tree, return None
        if self._tree.GetListOfLeaves().FindObject("du_id") == None:
            return None

        # Try to store the currently read entry
        try:
            current_entry = self._tree.GetReadEntry()
        # if failed, store None
        except:
            current_entry = None

        count = self.draw("du_id", "", "goff")
        detector_units = np.unique(np.array(np.frombuffer(self.get_v1(), dtype=np.float64, count=count)).astype(int))

        # Get the detector units branch
        # It has to be here, not before the draw(), due to a bug in PyROOT
        du_br = self._tree.GetBranch("du_id")

        # If there was an entry read before this action, come back to this entry
        if current_entry is not None:
            du_br.GetEntry(current_entry)

        return detector_units

    def get_list_of_all_used_dus(self):
        """Compiles the list of all detector units used in the events of the tree"""
        dus = self.get_list_of_dus()
        if dus is not None:
            return np.unique(np.array(dus).flatten()).tolist()
        else:
            return None

    def get_dus_indices_in_run(self, trun):
        """Gets an array of the indices of DUs of the current event in the TRun tree"""

        return np.nonzero(np.isin(np.asarray(trun.du_id), np.asarray(self.du_id)))[0]


@dataclass
## The class for storing ADC traces and associated values for each event
class TADC(MotherEventTree):
    """The class for storing ADC traces and associated values for each event"""

    _type: str = "adc"

    _tree_name: str = "tadc"

    ## Common for the whole event
    ## Event size
    """Common for the whole event"""
    event_size: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    ## Event in the run number
    t3_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event in the run number"""
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the first triggered station
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the first triggered station"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Event format version of the DAQ
    event_version: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event format version of the DAQ"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## The T3 trigger number
    event_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """The T3 trigger number"""
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    ## Atmospheric temperature (read via I2C)
    atm_temperature: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric temperature (read via I2C)"""
    ## Atmospheric pressure
    atm_pressure: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric pressure"""
    ## Atmospheric humidity
    atm_humidity: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Atmospheric humidity"""
    ## Acceleration of the antenna in X
    acceleration_x: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in X"""
    ## Acceleration of the antenna in Y
    acceleration_y: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in Y"""
    ## Acceleration of the antenna in Z
    acceleration_z: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Acceleration of the antenna in Z"""
    ## Battery voltage
    battery_level: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Battery voltage"""
    ## Firmware version
    firmware_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Firmware version"""
    ## ADC sampling frequency in MHz
    adc_sampling_frequency: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling frequency in MHz"""
    ## ADC sampling resolution in bits
    adc_sampling_resolution: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADC sampling resolution in bits"""

    adc_input_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """ADC input channels"""

    adc_enabled_channels_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """ADC enabled channels"""

    adc_samples_count_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    trigger_pattern_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    trigger_pattern_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    trigger_pattern_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    clock_tick: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock tick at which the event was triggered (used to calculate the trigger time)"""
    ## Clock ticks per second
    clock_ticks_per_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock ticks per second"""
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    gps_offset: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
    ## GPS leap second
    gps_leap_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS leap second"""
    ## GPS status
    gps_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS status"""
    ## GPS alarms
    gps_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS alarms"""
    ## GPS warnings
    gps_warnings: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS warnings"""
    ## GPS time
    gps_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS time"""
    ## Longitude
    gps_long: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Longitude"""
    ## Latitude
    gps_lat: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Latitude"""
    ## Altitude
    gps_alt: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Altitude"""
    ## GPS temperature
    gps_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS temperature"""

    ## Digital control register
    enable_auto_reset_timeout: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Digital control register"""
    force_firmware_reset: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_filter_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    enable_1PPS: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_DAQ: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Trigger enable mask register
    enable_trigger_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Trigger enable mask register"""
    enable_trigger_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    enable_trigger_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))

    ## Test pulse rate divider and channel readout enable
    enable_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Test pulse rate divider and channel readout enable"""
    fire_single_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    test_pulse_rate_divider: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))

    ## Common coincidence readout time window
    common_coincidence_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Common coincidence readout time window"""

    ## Input selector for readout channel
    selector_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Input selector for readout channel"""

    pre_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    post_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    gain_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    integration_time_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    offset_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    base_maximum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    base_minimum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))

    signal_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    noise_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    tper_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    tprev_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    ncmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    tcmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    qmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    ncmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    qmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))

    ## ?? What is it? Some kind of the adc trace offset?
    ioff: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """?? What is it? Some kind of the adc trace offset?"""

    ## ADC traces for channels (0,1,2,3)
    trace_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<short>>"))
    """ADC traces for channels (0,1,2,3)"""

    ## PPS-ID
    pps_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """PPS-ID"""

    ## FPGA temperature
    fpga_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """FPGA temperature"""

    ## ADC temperature
    adc_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """ADC temperature"""

    ## Hardware ID
    hardware_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Hardware ID"""

    ## Trigger status
    trigger_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger status"""

    ## Trigger DDR storage
    trigger_ddr_storage: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger DDR storage"""

    ## Data format version
    data_format_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Data format version"""

    ## ADAQ version
    adaq_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """ADAQ version"""

    ## DUDAQ version
    dudaq_version: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """DUDAQ version"""

    ## Trigger selection: ch0&ch1&ch2
    trigger_pattern_ch0_ch1_ch2: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: ch0&ch1&ch2"""

    ## Trigger selection: ch0&ch1&~ch2
    trigger_pattern_ch0_ch1_notch2: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: ch0&ch1&~ch2"""

    ## Trigger selection: 20 Hz
    trigger_pattern_20Hz: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Trigger selection: 20 Hz"""

    ## External pulse trigger period
    trigger_external_test_pulse_period: StdVectorListDesc = field(default=StdVectorListDesc("int"))
    """External pulse trigger period"""

    ## GPS seconds since Sunday 00:00
    gps_sec_sun: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS seconds since Sunday 00:00"""

    ## GPS week number
    gps_week_num: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS week number"""

    ## GPS receiver mode
    gps_receiver_mode: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS receiver mode"""

    ## GPS disciplining mode
    gps_disciplining_mode: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS disciplining mode"""

    ## GPS self-survey progress
    gps_self_survey: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS self-survey progress"""

    ## GPS minor alarms
    gps_minor_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS minor alarms"""

    ## GPS GNSS decoding
    gps_gnss_decoding: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """GPS GNSS decoding"""

    ## GPS disciplining activity
    gps_disciplining_activity: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS disciplining activity"""

    ## Notch filter number
    notch_filters_no_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Notch filter number"""


@dataclass
## The class for storing voltage traces and associated values for each event
class TRawVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at ADC input level for each event. Derived from TADC but in human readable format and physics units."""

    _type: str = "rawvoltage"

    _tree_name: str = "trawvoltage"
    ## Common for the whole event
    ## Event size
    """Common for the whole event"""
    event_size: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    ## Trigger position in the trace, in samples
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""
    ## Atmospheric temperature (read via I2C)
    atm_temperature: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric temperature (read via I2C)"""
    ## Atmospheric pressure
    atm_pressure: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric pressure"""
    ## Atmospheric humidity
    atm_humidity: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Atmospheric humidity"""
    ## Acceleration of the antenna in (x,y,z) in m/s2
    du_acceleration: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Acceleration of the antenna in (x,y,z) in m/s2"""
    ## Battery voltage
    battery_level: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Battery voltage"""
    ## ADC samples callected in channels (0,1,2,3)
    adc_samples_count_channel: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """ADC samples callected in channels (0,1,2,3)"""
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    trigger_pattern: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?"""
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    clock_tick: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock tick at which the event was triggered (used to calculate the trigger time)"""
    ## Clock ticks per second
    clock_ticks_per_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock ticks per second"""
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    gps_offset: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
    ## GPS leap second
    gps_leap_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS leap second"""
    ## GPS status
    gps_status: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS status"""
    ## GPS alarms
    gps_alarms: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS alarms"""
    ## GPS warnings
    gps_warnings: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """GPS warnings"""
    ## GPS time
    gps_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS time"""
    ## Longitude
    gps_long: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Longitude"""
    ## Latitude
    gps_lat: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Latitude"""
    ## Altitude
    gps_alt: StdVectorListDesc = field(default=StdVectorListDesc("double"))
    """Altitude"""
    ## GPS temperature
    gps_temp: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """GPS temperature"""

    ## ?? What is it? Some kind of the adc trace offset?
    ioff: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """?? What is it? Some kind of the adc trace offset?"""

    ## Voltage traces for channels 1,2,3,4 in muV
    trace_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Voltage traces for channels 1,2,3,4 in muV"""


@dataclass
## The class for storing voltage traces and associated values for each event
class TVoltage(MotherEventTree):
    """The class for storing voltage traces and associated values at antenna feed point for each event"""

    _type: str = "voltage"

    _tree_name: str = "tvoltage"

    ## Common for the whole event
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """First detector unit that triggered in the event"""
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Same as event_type, but event_type could consist of different triggered DUs"""
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""

    ## Acceleration of the antenna in (x,y,z) in m/s2
    du_acceleration: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Acceleration of the antenna in (x,y,z) in m/s2"""
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger rate - the number of triggers recorded in the second preceding the event"""

    ## Voltage traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Voltage traces for antenna arms (x,y,z)"""
    # _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<Float32_t>>"))

    ## Peak2peak amplitude (muV)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak2peak amplitude (muV)"""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """(Computed) peak time"""


@dataclass
## The class for storing Efield traces and associated values for each event
class TEfield(MotherEventTree):
    """The class for storing Efield traces and associated values for each event"""

    _type: str = "efield"

    _tree_name: str = "tefield"

    ## Common for the whole event
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Unix time corresponding to the GPS seconds of the trigger"""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """GPS nanoseconds corresponding to the trigger of the first triggered station"""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detector units in the event - basically the antennas count"""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Detector unit (antenna) ID"""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Unix time of the trigger for this DU"""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Nanoseconds of the trigger for this DU"""

    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger position in the trace, in samples"""


    ## Efield traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Efield traces for antenna arms (x,y,z)"""
    ## FFT magnitude for antenna arms (x,y,z)
    fft_mag: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """FFT magnitude for antenna arms (x,y,z)"""
    ## FFT phase for antenna arms (x,y,z)
    fft_phase: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """FFT phase for antenna arms (x,y,z)"""

    ## Peak-to-peak amplitudes for X, Y, Z (muV/m)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak-to-peak amplitudes for X, Y, Z (muV/m)"""
    ## Efield polarisation info
    pol: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Efield polarisation info"""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """(Computed) peak time"""


@dataclass
## The class for storing reconstructed shower data common for each event
class TShower(MotherEventTree):
    """The class for storing shower data common for each event"""

    _type: str = "shower"

    _tree_name: str = "tshower"

    ## Shower primary type
    primary_type: StdStringDesc = field(default=StdStringDesc(""))
    """Shower primary type"""
    ## Energy from e+- (ie related to radio emission) (GeV)
    energy_em: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Energy from e+- (ie related to radio emission) (GeV)"""
    ## Total energy of the primary (including muons, neutrinos, ...) (GeV)
    energy_primary: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Total energy of the primary (including muons, neutrinos, ...) (GeV)"""
    ## Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")
    azimuth: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")"""
    ## Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")
    zenith: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")"""
    ## Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref
    direction: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref"""
    ## Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)
    shower_core_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)"""
    ## Atmospheric model name
    atmos_model: StdStringDesc = field(default=StdStringDesc(""))
    """Atmospheric model name"""
    ## Atmospheric model parameters
    atmos_model_param: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Atmospheric model parameters"""
    ## Magnetic field parameters: Inclination, Declination, modulus
    magnetic_field: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Magnetic field parameters: Inclination, Declination, modulus"""
    ## Ground Altitude at core position (m asl)
    core_alt: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Ground Altitude at core position (m asl)"""
    ## Shower Xmax depth  (g/cm2 along the shower axis)
    xmax_grams: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Shower Xmax depth  (g/cm2 along the shower axis)"""
    ## Shower Xmax position in GRAND detector ref
    xmax_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower Xmax position in GRAND detector ref"""
    ## Shower Xmax position in shower coordinates
    xmax_pos_shc: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Shower Xmax position in shower coordinates"""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Unix time when the shower was at the core position (seconds after epoch)"""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_ns: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Unix time when the shower was at the core position (seconds after epoch)"""


@dataclass
## The class for storing a shower sim-only data for each event
class TShowerSim(MotherEventTree):
    """Event-level info associated with simulated showers"""

    _type: str = "showersim"

    _tree_name: str = "tshowersim"

    ## File name in the simulator
    input_name: StdStringDesc = field(default=StdStringDesc())
    """File name in the simulator"""
    ## The date for which we simulate the event (epoch)
    event_date: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """The date for which we simulate the event (epoch)"""
    ## Random seed
    rnd_seed: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Random seed"""
    ## Primary energy (GeV)
    # primary_energy: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Primary particle type
    # primary_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))
    ## Primary injection point in Shower Coordinates
    primary_inj_point_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Primary injection point in Shower Coordinates"""
    ## Primary injection altitude in Shower Coordinates
    primary_inj_alt_shc: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Primary injection altitude in Shower Coordinates"""
    ## Primary injection direction in Shower Coordinates
    primary_inj_dir_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Primary injection direction in Shower Coordinates"""

    ## Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    atmos_altitude: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]"""
    atmos_density: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_depth: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ## High energy hadronic model (and version) used
    hadronic_model: StdStringDesc = field(default=StdStringDesc())
    """High energy hadronic model (and version) used"""
    ## Energy model (and version) used
    low_energy_model: StdStringDesc = field(default=StdStringDesc())
    """Energy model (and version) used"""
    ## Time it took for the sim
    cpu_time: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Time it took for the sim"""

    ## Slant depth of the observing levels for longitudinal development tables
    long_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Slant depth of the observing levels for longitudinal development tables"""
    long_pd_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Number of electrons
    long_pd_eminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of electrons"""
    ## Number of positrons
    long_pd_eplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of positrons"""
    ## Number of muons-
    long_pd_muminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of muons-"""
    ## Number of muons+
    long_pd_muplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of muons+"""
    ## Number of gammas
    long_pd_gamma: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of gammas"""
    ## Number of pions, kaons, etc.
    long_pd_hadron: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Number of pions, kaons, etc."""
    ## Energy in low energy gammas
    long_gamma_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy gammas"""
    ## Energy in low energy e+/e-
    long_e_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy e+/e-"""
    ## Energy deposited by e+/e-
    long_e_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by e+/e-"""
    ## Energy in low energy mu+/mu-
    long_mu_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy mu+/mu-"""
    ## Energy deposited by mu+/mu-
    long_mu_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by mu+/mu-"""
    ## Energy in low energy hadrons
    long_hadron_elow: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in low energy hadrons"""
    ## Energy deposited by hadrons
    long_hadron_edep: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy deposited by hadrons"""
    ## Energy in created neutrinos
    long_neutrino: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Energy in created neutrinos"""
    ## Core positions tested for that shower to generate the event (effective area study)
    tested_core_positions: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Core positions tested for that shower to generate the event (effective area study)"""

    event_weight: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """statistical weight given to the event"""
    tested_cores: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """tested core positions"""

@dataclass
class TRecons(MotherEventTree):
    """TTree to store reconstruction results per event"""

    _type: str = "recons"
    _tree_name: str = "trecons"

    # Event identifiers
    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    event_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    
    #Event processing
    ## Maximum amplitude of the Hilbert envelope
    ## (in ADC counts or µV/m depending on the input)
    peak_amps:  StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Time corresponding to the peak amplitude (in seconds)
    peak_time: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Antenna positions in the GRAND detector reference frame (in meters)
    Xants:  StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    ## Number of triggered antennas
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))

    # Plane Wave Fits (PWF) reconstruction outputs 
    ## Shower zenith angle from PWF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    zenith_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Shower azimuth angle from PWF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    azimuth_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Non-reduced chi² from PWF
    ## Divide by du_count to obtain the reduced chi²
    chi2_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))

    # Spherical Wave Fits (SWF) Reconstruction outputs 
    ## polar zenith from SWF (in rad) 
    zenith_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## polar azimuth from SWF (in rad)
    azimuth_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Distance between the reconstructed Xsource and the origin = layout center (in meters)
    r_xmax: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Emission time from SWF (in seconds)
    t_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Reconstructed shower emission point (Xsource) in GRAND detector frame (in meters)
    ## Coordinate system: NWU, origin at layout center
    ## x = r_xmax * sin(theta_swf) * cos(phi_swf)
    ## y = r_xmax * sin(theta_swf) * sin(phi_swf)
    ## z = r_xmax * cos(theta_swf)
    Xsource: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    ## Non-reduced chi² from SWF
    ## Divide by du_count to obtain the reduced chi²
    chi2_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Distance between the reconstructed Xsource and each antenna (in meters)
    distance_source_antenna:  StdVectorListDesc = field(default=StdVectorListDesc("float"))

    # Angular Distribution Function (ADF)
    ## Shower zenith angle from ADF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    zenith_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Shower azimuth angle from ADF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    azimuth_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Width parameter from ADF fit (delta_omega)
    width: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Scaling factor A from ADF fit
    scaling_factor: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Non-reduced chi² from ADF
    ## Divide by du_count to obtain the reduced chi²
    chi2_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Azimuth angle in the shower plane (in radians)
    eta: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Angular distance to the shower axis (in radians)
    omega:  StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Cherenkov angle (computed using a two emission points toy model) (in radians)
    omega_cr:  StdVectorListDesc = field(default=StdVectorListDesc("float"))
    # Amplitude predicted by the ADF model at each antenna
    ## (in ADC counts or µV/m)
    adf_amplitude:  StdVectorListDesc = field(default=StdVectorListDesc("float"))

    ## (Electromagnetic energy in eV, obtained directly from voltage data)
    energy_elm_voltage:  TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))


    ## Cramér-Rao (lower) bound (CRB)
    ## CRB of shower zenith angle from ADF (in radians)
    crb_zenith_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower azimuth angle from ADF (in radians)
    crb_azimuth_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of scaling factor A from ADF fit
    crb_scaling_factor: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of width parameter from ADF fit (delta_omega)
    crb_width: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower zenith from SWF (in rad)
    crb_zenith_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower azimuth from SWF (in rad)
    crb_azimuth_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of distance between the reconstructed Xsource and the origin = layout center (in meters)
    crb_r_xmax: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of emission time from SWF (in seconds)
    crb_t_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower zenith from PWF (in radians)
    crb_zenith_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower azimuth from PWF (in radians)
    crb_azimuth_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))

