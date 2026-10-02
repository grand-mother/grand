# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field

import numpy as np
import ROOT

from grand.dataio import DataTree, TTreeScalarDesc, NotUniqueEvent, grand_tree_list, TRun, logger, StdVectorListDesc, StdStringDesc, TTreeArrayDesc
from grand.dataio import file_lock as _file_lock
from grand.basis import validate as _validate


@dataclass
## A mother class for classes with Event values
class MotherEventTree(DataTree):
    """A mother class for classes with Event values"""

    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run number; with the event number it identifies the entry"""
    # ToDo: it seems instances propagate this number among them without setting (but not the run number!). I should find why...
    event_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event number, unique within its run"""

    def __post_init__(self):
        r"""Completes initialization after the dataclass fields are set.

        """
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
        self._check_open("fill")
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(
                f"An event with (run_number,event_number)=({self.run_number},{self.event_number}) already exists in the TTree {self._tree.GetName()}."
            )

        # Repoen the file in write mode, if it exists
        # Reopening in case of different mode takes here ~0.06 s, in case of the same mode, 0.0005 s, so negligible
        if self._file is not None:
            # One writer at a time, with an up-to-date view of the file: a file
            # reopened for update is rewritten when it is closed (#281)
            _file_lock.lock_for_writing(self._file.GetName(), type(self).__name__)
            self._file.ReOpen("update")

        # Fill the tree
        self._tree.Fill()
        # Held until written: dropping it now would lose this entry (#284)
        grand_tree_list.pin(self)

        # If there is no entry list, create it
        if not self._entry_list:
            self.fill_entry_list()
        # Add the current run_number and event_number to the entry_list
        self._entry_list.append((self.run_number, self.event_number))

    def add_proper_friends(self):
        """Add proper friends to this tree

        Returns
        -------
        None
            Attaches the run and simulation trees this one needs.
        """
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
        count = self.draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        print("List of events in the tree:")
        print("event_number run_number")
        for i in range(count):
            print(int(events[i]), int(runs[i]))

    ## Gets list of events in the tree together with runs
    def get_list_of_events(self):
        """Gets list of events in the tree together with runs

        Returns
        -------
        list of tuple
            Every ``(event number, run number)`` in the tree.
        """
        count = self.draw("event_number:run_number", "", "goff")
        events = self._tree.GetV1()
        runs = self._tree.GetV2()
        return [(int(events[i]), int(runs[i])) for i in range(count)]

    ## Readout the TTree entry corresponding to the event and run
    def get_event(self, ev_no, run_no=0):
        """Readout the TTree entry corresponding to the event and run

        Parameters
        ----------
        ev_no : int
            Event number.
        run_no : int, optional
            Run number; the pair is unique.

        Returns
        -------
        int
            Bytes read.

        Raises
        ------
        LookupError
            When the tree has no such event.
        """
        self._check_open("get_event")
        # Try to get the requested entry
        # res = self._tree.GetEntryWithIndex(int(run_no), int(ev_no))
        # The above should work, but there is a bug in ROOT
        # int() gave a bare "invalid literal for int()" for get_event('x') (#236)
        ev_no = self._integer(ev_no, "get_event", "ev_no")
        run_no = self._integer(run_no, "get_event", "run_no")
        self._current_index("run_number", "event_number")
        entry = self._tree.GetEntryNumberWithIndex(run_no, ev_no)
        res = self._tree.GetEntry(entry) if entry >= 0 else 0
        if res <= 0:
            raise LookupError(_validate.message(
                type(self).__name__, "get_event: no event %d in run %d in the %s "
                "tree" % (ev_no, run_no, self.tree_name)))

        self.assign_branches()

        return res

    ## Check if the TTree has an entry with the given event and run number
    def has_event(self, ev_no, run_no=0):
        """Check if the TTree has an entry with the given event and run number

        Parameters
        ----------
        ev_no : int
            Event number.
        run_no : int, optional
            Run number; the pair is unique.

        Returns
        -------
        bool
            True when the tree holds that event.
        """
        self._check_open("has_event")
        ev_no = self._integer(ev_no, "has_event", "ev_no")
        run_no = self._integer(run_no, "has_event", "run_no")
        self._current_index("run_number", "event_number")
        return self._tree.GetEntryNumberWithIndex(run_no, ev_no) >= 0

    ## Builds index based on run_id and evt_id for the TTree
    def build_index(self, run_id, evt_id):
        """Builds index based on run_id and evt_id for the TTree

        Parameters
        ----------
        run_id : str, optional
            Branch holding the run number.
        evt_id : str, optional
            Branch holding the event number.
        """
        self._check_open("build_index")
        self._reset_read_cache(self._tree)
        self._tree.BuildIndex(run_id, evt_id)

    ## Fills the entry list from the tree
    def fill_entry_list(self, tree=None):
        """Fills the entry list from the tree

        Parameters
        ----------
        tree : ROOT.TTree, optional
            Tree to index; this one by default.
        """
        if tree is None:
            tree = self._tree
        # Fill the entry list if there are some entries in the tree.  The
        # cache reset matters here: this runs on a tree that is being
        # appended to, every time it is reopened (issue #89).
        self._reset_read_cache(tree)
        with self._kept_buffers("run_number:event_number"):
            count = tree.Draw("run_number:event_number", "", "goff")
        if count > 0:
            v1 = np.array(np.frombuffer(tree.GetV1(), dtype=np.float64, count=count)).astype(int)
            v2 = np.array(np.frombuffer(tree.GetV2(), dtype=np.float64, count=count)).astype(int)
            self._entry_list = [(int(el[0]), int(el[1])) for el in zip(v1, v2)]
        # Remove the Draw() generated histogram from current file to prevent saving
        if tmph := ROOT.gDirectory.Get("htemp"):
            tmph.SetDirectory(0)

    ## Check if specified run_number/event_number already exist in the tree
    def is_unique_event(self):
        """Check if specified run_number/event_number already exist in the tree

        Returns
        -------
        bool
            True when no ``(event, run)`` pair appears twice.
        """
        # If there is no entry list, create it
        if not self._entry_list:
            self.fill_entry_list()
        # If the entry list does not exist, the event is unique
        if self._entry_list and (self.run_number, self.event_number) in self._entry_list:
            return False

        return True

    def get_traces_lengths(self):
        """Gets the trace lengths of the current entry

        Returns
        -------
        list of list of int or None
            For each detection unit of the loaded entry, the length of each of
            its channels; ``None`` if this tree holds no traces.  (It looked
            for branches named ``trace_x`` or ``trace_0``, which no tree has,
            and always returned ``None``.)
        """
        for name in ("trace", "trace_ch"):
            if self._tree.GetListOfLeaves().FindObject(name):
                return [[len(channel) for channel in du] for du in getattr(self, name)]
        return None

    def get_list_of_dus(self):
        """Gets the detector units of the current entry

        Returns
        -------
        list of int or None
            Detection units in the loaded entry, in its order; ``None`` if this
            tree has no ``du_id``.  For the units of the whole tree, use
            `get_list_of_all_used_dus`.
        """
        if not self._tree.GetListOfLeaves().FindObject("du_id"):
            return None
        return [int(du) for du in self.du_id]

    def get_list_of_all_used_dus(self):
        """Compiles the list of all detector units used in the events of the tree

        Returns
        -------
        list of int or None
            Every detection unit appearing anywhere in the tree, sorted;
            ``None`` if this tree has no ``du_id``.
        """
        if not self._tree.GetListOfLeaves().FindObject("du_id"):
            return None
        # draw() keeps the loaded entry's du_id (#196)
        count = self.draw("du_id", "", "goff")
        return np.unique(np.frombuffer(self.get_v1(), dtype=np.float64, count=count).astype(int)).tolist()

    def get_dus_indices_in_run(self, trun):
        """Gets an array of the indices of DUs of the current event in the TRun tree

        Parameters
        ----------
        trun : TRun
            The run tree to look the units up in.

        Returns
        -------
        ndarray
            Index of each unit of this event within the run unit list, in the
            event's order, so that ``run_array[indices]`` lines up with this
            event's traces.

        Raises
        ------
        ValueError
            If a unit of this event is not in the run.
        """
        # In the event's order: this returned the matches in the run's order,
        # pairing positions and sampling times with the wrong traces whenever
        # the two orders differ, and dropped units missing from the run (#199)
        index = {int(du): i for i, du in enumerate(trun.du_id)}
        event_dus = [int(du) for du in self.du_id]
        missing = [du for du in event_dus if du not in index]
        if missing:
            raise ValueError(_validate.message(
                "%s.get_dus_indices_in_run" % type(self).__name__,
                "units %s of event %s (run %s) are not in the run's du_id"
                % (missing, self.event_number, self.run_number)))
        return np.array([index[du] for du in event_dus], dtype=int)


@dataclass
## The class for storing ADC traces and associated values for each event
class TADC(MotherEventTree):
    """Digitized traces of each event, in ADC counts, with each detection unit's status and firmware settings."""

    _type: str = "adc"

    _tree_name: str = "tadc"

    ## Common for the whole event
    ## Event size
    event_size: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Size of the event as recorded by the DAQ."""
    ## Event in the run number
    t3_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of the T3 (array-level) trigger that recorded the event."""
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Identifier of the first unit that triggered."""
    ## Unix time corresponding to the GPS seconds of the first triggered station
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Time of the event's trigger, in Unix seconds, converted from GPS time."""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Nanoseconds to add to ``time_seconds``, for the first unit that triggered."""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type: 0x1000 for the 10-second trigger, 0x8000 for a random trigger; any other value is a shower trigger."""
    ## Event format version of the DAQ
    event_version: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Version of the DAQ's event format."""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detection units in the event."""

    ## Specific for each Detector Unit
    ## The T3 trigger number
    event_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Number of the T3 trigger, for each unit."""
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Identifier of each detection unit, in the order of the traces."""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Trigger time of each unit, in Unix seconds."""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int", maximum=999999999, unit="ns"))
    """Nanoseconds to add to ``du_seconds``, for each unit."""
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Sample of each unit's trace at which it triggered."""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger type of each unit, with the codes of ``event_type``; for simulations with the offline T1 trigger, 1 if the unit passed and 0 if not."""
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
    """Number of samples recorded on each channel, for each unit."""

    trigger_pattern_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Which channels triggered, for each unit."""
    trigger_pattern_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the coincidence of channels 0 and 1 triggered the unit."""
    trigger_pattern_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether channel 1 without channel 0 triggered the unit."""
    trigger_pattern_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the reduced coincidence of channels 0 and 1 triggered the unit."""
    trigger_pattern_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the coincidence of channels 2 and 3 triggered the unit."""
    trigger_pattern_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether a calibration trigger recorded the unit."""
    trigger_pattern_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the 10-second trigger recorded the unit."""
    trigger_pattern_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether an external test pulse triggered the unit."""

    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Number of triggers each unit recorded in the second before the event."""
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    clock_tick: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock tick at which the event was triggered (used to calculate the trigger time)"""
    ## Clock ticks per second
    clock_ticks_per_second: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Clock ticks per second"""
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    gps_offset: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """Offset between the GPS one-pulse-per-second signal and the true second."""
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
    """Longitude of each unit, as reported by its GPS receiver."""
    ## Latitude
    gps_lat: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Latitude of each unit, as reported by its GPS receiver."""
    ## Altitude
    gps_alt: StdVectorListDesc = field(default=StdVectorListDesc("unsigned long long"))
    """Altitude of each unit, as reported by its GPS receiver."""
    ## GPS temperature
    gps_temp: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """GPS temperature"""

    ## Digital control register
    enable_auto_reset_timeout: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Digital control register"""
    force_firmware_reset: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Firmware control register: force a reset."""
    enable_filter_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Whether each channel's digital filter is on, for each unit."""
    enable_1PPS: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Firmware control register: the GPS one-pulse-per-second signal is on."""
    enable_DAQ: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Firmware control register: data taking is on."""

    ## Trigger enable mask register
    enable_trigger_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Trigger enable mask register"""
    enable_trigger_ch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the coincidence of channels 0 and 1 may trigger the unit."""
    enable_trigger_notch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether channel 1 without channel 0 may trigger the unit."""
    enable_trigger_redch0_ch1: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the reduced coincidence of channels 0 and 1 may trigger the unit."""
    enable_trigger_ch2_ch3: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the coincidence of channels 2 and 3 may trigger the unit."""
    enable_trigger_calibration: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether calibration triggers are on."""
    enable_trigger_10s: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether the 10-second trigger is on."""
    enable_trigger_external_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Whether an external test pulse may trigger the unit."""

    ## Test pulse rate divider and channel readout enable
    enable_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<bool>"))
    """Test pulse rate divider and channel readout enable"""
    fire_single_test_pulse: StdVectorListDesc = field(default=StdVectorListDesc("bool"))
    """Firmware control register: fire one test pulse."""
    test_pulse_rate_divider: StdVectorListDesc = field(default=StdVectorListDesc("unsigned char"))
    """Divider that sets the test-pulse rate."""

    ## Common coincidence readout time window
    common_coincidence_time: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Common coincidence readout time window"""

    ## Input selector for readout channel
    selector_readout_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Input selector for readout channel"""

    pre_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Readout window before the coincidence, per channel, as set in the unit's firmware."""
    post_coincidence_window_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Readout window after the coincidence, per channel, as set in the unit's firmware."""

    gain_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Gain correction, per channel, as set in the unit's firmware."""
    integration_time_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """Integration time, per channel, as set in the unit's firmware."""
    offset_correction_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Offset correction, per channel, as set in the unit's firmware."""
    base_maximum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Upper limit of the baseline, per channel, as set in the unit's firmware."""
    base_minimum_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """Lower limit of the baseline, per channel, as set in the unit's firmware."""

    signal_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """T1 threshold (``th1`` in :mod:`grand.sim.detector.trigger`), per channel, as set in the unit's firmware."""
    noise_threshold_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>"))
    """T2 threshold (``th2``), per channel, as set in the unit's firmware."""
    tper_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """T1 window length (``t_period``), per channel, as set in the unit's firmware."""
    tprev_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """Quiet time before the T1 crossing (``t_quiet``), per channel, as set in the unit's firmware."""
    ncmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """Largest number of T2 crossings (``nc_max``), per channel, as set in the unit's firmware."""
    tcmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """Largest separation between T2 crossings (``t_sepmax``), per channel, as set in the unit's firmware."""
    qmax_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Upper limit of the charge ratio (``q_max``), per channel, as set in the unit's firmware."""
    ncmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned short>", "vector<unsigned char>"))
    """Smallest number of T2 crossings (``nc_min``), per channel, as set in the unit's firmware."""
    qmin_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<unsigned char>"))
    """Lower limit of the charge ratio (``q_min``), per channel, as set in the unit's firmware."""

    ## ?? What is it? Some kind of the adc trace offset?
    ioff: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Offset of the ADC trace, for each unit; its exact meaning is not documented."""

    ## ADC traces for channels (0,1,2,3)
    trace_ch: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<short>>"))
    """ADC counts of each unit's channels 0 to 3: one row per unit, one trace per channel."""

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

    ## NUTRIG correlation with X
    nutrig_rhox: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """NUTRIG correlation with X"""

    ## NUTRIG correlation with Y
    nutrig_rhoy: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """NUTRIG correlation with Y"""

@dataclass
## The class for storing voltage traces and associated values for each event
class TRawVoltage(MotherEventTree):
    """Voltage at the ADC input of each event, converted from ``TADC`` to physical units."""

    _type: str = "rawvoltage"

    _tree_name: str = "trawvoltage"
    ## Common for the whole event
    ## Event size
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
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int", maximum=999999999, unit="ns"))
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

    ## NUTRIG correlation with X
    nutrig_rhox: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """NUTRIG correlation with X"""

    ## NUTRIG correlation with Y
    nutrig_rhoy: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    """NUTRIG correlation with Y"""


@dataclass
## The class for storing voltage traces and associated values for each event
class TVoltage(MotherEventTree):
    """Voltage traces of each event, in µV, as simulated by ``Efield2Voltage``."""

    _type: str = "voltage"

    _tree_name: str = "tvoltage"

    ## Common for the whole event
    ## First detector unit that triggered in the event
    first_du: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Identifier of the first unit that triggered."""
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Time of the event's trigger, in Unix seconds, converted from GPS time."""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Nanoseconds to add to ``time_seconds``, for the first unit that triggered."""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detection units in the event."""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Identifier of each detection unit, in the order of the traces."""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Trigger time of each unit, in Unix seconds."""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int", maximum=999999999, unit="ns"))
    """Nanoseconds to add to ``du_seconds``, for each unit."""
    ## Same as event_type, but event_type could consist of different triggered DUs
    trigger_flag: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Trigger type of each unit, with the codes of ``event_type``."""
    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Sample of each unit's trace at which it triggered."""

    ## Acceleration of the antenna in (x,y,z) in m/s2
    du_acceleration: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Acceleration measured at each unit, as (x, y, z), in m/s²."""
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    trigger_rate: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Number of triggers each unit recorded in the second before the event."""

    ## Voltage traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Voltage at each unit, in µV: one row per unit, holding the X (south-north), Y (east-west) and Z (vertical) arms.  A simulated voltage is at the ADC input, or at the antenna terminals when the RF chain is left out."""
    # _trace: StdVectorList = field(default_factory=lambda: StdVectorList("vector<vector<Float32_t>>"))

    ## Peak2peak amplitude (muV)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak-to-peak amplitude of each arm, for each unit, in µV."""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Time of the peak of each component, for each unit, in ns."""

    ## Version of GRANDlib that produced this file
    grandlib_version: StdStringDesc = field(default=StdStringDesc())
    r"""Version of GRANDlib that produced this file.

    Written by :class:`~grand.sim.efield2voltage.Efield2Voltage`.  The
    simulated voltage depends on the code as well as on the input: the
    Galactic-noise level, for example, changed by a factor of :math:`\sqrt2`
    on 7 September 2026.  Files written before that date carry no version.
    It sat at ``0.1.0.dev0`` across twenty-seven milestone tags, which would
    have made the stamp useless; see the note in ``pyproject.toml``.
    """


@dataclass
## The class for storing Efield traces and associated values for each event
class TEfield(MotherEventTree):
    """Electric-field traces of each event at each detection unit, in µV/m.

    Examples
    --------
    Read the electric-field traces of one event, one row per detection unit:

    .. jupyter-execute::

        from pathlib import Path

        import numpy as np
        import grand
        from grand.dataio import TEfield

        sample = (Path(grand.__file__).parents[1]
                  / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
        with TEfield(str(sample / "efield_1618-13790_L0_0000.root")) as tefield:
            tefield.get_event(13790, 1)
            traces = np.asarray(tefield.trace)               # (units, 3, samples), µV/m
            print("%d units, peak %.0f µV/m" % (len(tefield.du_id), np.abs(traces).max()))
    """

    _type: str = "efield"

    _tree_name: str = "tefield"

    ## Common for the whole event
    ## Unix time corresponding to the GPS seconds of the trigger
    time_seconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Time of the event's trigger, in Unix seconds, converted from GPS time."""
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    time_nanoseconds: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Nanoseconds to add to ``time_seconds``, for the first unit that triggered."""
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    event_type: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Trigger type: 0x1000 for the 10-second trigger, 0x8000 for a random trigger; any other value is a shower trigger."""
    ## Number of detector units in the event - basically the antennas count
    du_count: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Number of detection units in the event."""

    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    du_id: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short", "unsigned int"))
    """Identifier of each detection unit, in the order of the traces."""
    ## Unix time of the trigger for this DU
    du_seconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int"))
    """Trigger time of each unit, in Unix seconds."""
    ## Nanoseconds of the trigger for this DU
    du_nanoseconds: StdVectorListDesc = field(default=StdVectorListDesc("unsigned int", maximum=999999999, unit="ns"))
    """Nanoseconds to add to ``du_seconds``, for each unit."""

    trigger_position: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))
    """Sample of each unit's trace at which it triggered."""


    ## Efield traces for antenna arms (x,y,z)
    trace: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Electric field at each unit, in µV/m: one row per unit, holding the x, y and z components."""
    ## FFT magnitude for antenna arms (x,y,z)
    fft_mag: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Magnitude of the Fourier transform of each component, for each unit."""
    ## FFT phase for antenna arms (x,y,z)
    fft_phase: StdVectorListDesc = field(default=StdVectorListDesc("vector<vector<float>>"))
    """Phase of the Fourier transform of each component, for each unit."""

    ## Peak-to-peak amplitudes for X, Y, Z (muV/m)
    p2p: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Peak-to-peak amplitude of each component, for each unit, in µV/m."""
    ## Efield polarization info
    pol: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Polarization of the field at each unit."""
    ## (Computed) peak time
    time_max: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Time of the peak of each component, for each unit, in ns."""


@dataclass
## The class for storing reconstructed shower data common for each event
class TShower(MotherEventTree):
    """The shower of each event: primary, energy, direction, core and shower maximum.

    Examples
    --------
    Read the shower of one event:

    .. jupyter-execute::

        from pathlib import Path

        import grand
        from grand.dataio import TShower

        sample = (Path(grand.__file__).parents[1]
                  / "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
        with TShower(str(sample / "shower_1618-13790_L0_0000.root")) as tshower:
            print("events (event, run):", tshower.get_list_of_events())
            tshower.get_event(13790, 1)
            print("zenith %.2f deg, azimuth %.2f deg, energy %.3g GeV"
                  % (tshower.zenith, tshower.azimuth, tshower.energy_primary))
    """

    _type: str = "shower"

    _tree_name: str = "tshower"

    ## Shower primary type
    primary_type: StdStringDesc = field(default=StdStringDesc(""))
    """Primary particle, as a PDG code written as text, such as ``"2212"`` for a proton."""
    ## Energy from e+- (ie related to radio emission) (GeV)
    energy_em: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, unit="GeV"))
    """Energy in the electromagnetic component of the shower (electrons and positrons), which produces the radio emission, in GeV."""
    ## Total energy of the primary (including muons, neutrinos, ...) (GeV)
    energy_primary: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, unit="GeV"))
    """Energy of the primary particle, in GeV."""
    ## Shower azimuth  (coordinates system = NWU + origin = core, "comes from")
    azimuth: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=360, unit="degrees"))
    """Azimuth of the direction the shower comes from, in degrees, measured from north toward west in the array frame (x north, y west, z up)."""
    ## Shower zenith  (coordinates system = NWU + origin = core, "comes from")
    zenith: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=180, unit="degrees"))
    """Zenith angle of the direction the shower comes from, in degrees: 0 for a vertical shower, above 90 for an upgoing one."""
    ## Direction vector (u_x, u_y, u_z)  of shower in GRAND detector ref
    direction: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Unit vector along which the shower travels, in the array frame (x north, y west, z up): minus the direction that ``zenith`` and ``azimuth`` name, so its z component is negative for a downgoing shower."""
    ## Shower core position in GRAND detector ref (if it is an upgoing shower, there is no core position)
    shower_core_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Position of the shower core, where the shower axis meets the ground, in meters in the array frame.  Not defined for an upgoing shower."""
    ## Atmospheric model name
    atmos_model: StdStringDesc = field(default=StdStringDesc(""))
    """Name of the atmospheric model the simulation used."""
    ## Atmospheric model parameters
    atmos_model_param: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Parameters of the atmospheric model; their meaning depends on the model and the simulation code."""
    ## Magnetic field parameters: Inclination, Declination, modulus
    magnetic_field: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Geomagnetic field the simulation used: inclination and declination in degrees, then the strength in µT.  Files from older CoREAS conversions hold the strength in mT or in gauss."""
    ## Ground Altitude at core position (m asl)
    core_alt: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    """Altitude of the ground at the core, in meters above sea level."""
    ## Shower Xmax depth  (g/cm2 along the shower axis)
    xmax_grams: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, unit="g/cm2"))
    """Atmospheric depth of the shower maximum, measured along the shower axis, in g/cm²."""
    ## Shower Xmax position in GRAND detector ref
    xmax_pos: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Position of the shower maximum, in meters in the array frame, the frame of ``du_xyz`` and ``shower_core_pos``.  NaN if unknown."""
    ## Shower Xmax position in shower coordinates
    xmax_pos_shc: TTreeArrayDesc = field(default=TTreeArrayDesc(3, np.float32))
    """Position of the shower maximum relative to the core, as the simulation gives it."""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Time at which the shower reached the core, in Unix seconds."""
    ## Unix time when the shower was at the core position (seconds after epoch)
    core_time_ns: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float64))
    """Nanoseconds to add to ``core_time_s``."""


@dataclass
## The class for storing a shower sim-only data for each event
class TShowerSim(MotherEventTree):
    """What a simulation records about the shower of each event, beyond ``TShower``."""

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
    """Air density at each altitude of ``atmos_altitude``, in g/cm3"""
    atmos_depth: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    """Vertical depth at each altitude of ``atmos_altitude``, in g/cm2"""

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
    """Slant depth of the levels of the particle-number profiles (``long_pd_*``), in g/cm2"""
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
    """Reconstruction results of each event: the fits of :mod:`grand.analysis`."""

    _type: str = "recons"
    _tree_name: str = "trecons"

    # Event identifiers
    run_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Run number; with the event number it identifies the entry"""
    event_number: TTreeScalarDesc = field(default=TTreeScalarDesc(np.uint32))
    """Event number, unique within its run"""

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
    ## The angles and their bounds are in radians: a value in degrees by
    ## mistake (85.0) warns, as it lies outside 0 to pi (#206)
    ## Shower zenith angle from PWF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    zenith_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## Shower azimuth angle from PWF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    azimuth_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=2 * np.pi, unit="radians"))
    ## Non-reduced (raw) chi² from PWF; NaN if not filled
    ## Divide by du_count - 2, the degrees of freedom, to obtain the reduced chi²
    chi2_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))

    # Spherical Wave Fits (SWF) Reconstruction outputs 
    ## polar zenith from SWF (in rad) 
    zenith_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## polar azimuth from SWF (in rad)
    azimuth_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=2 * np.pi, unit="radians"))
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
    ## Non-reduced (raw) chi² from SWF; NaN if not filled
    ## Divide by du_count - 4, the degrees of freedom, to obtain the reduced chi²
    chi2_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Distance between the reconstructed Xsource and each antenna (in meters)
    distance_source_antenna:  StdVectorListDesc = field(default=StdVectorListDesc("float"))

    # Angular Distribution Function (ADF)
    ## Shower zenith angle from ADF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    zenith_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## Shower azimuth angle from ADF (in radians)
    ## Coordinate system: NWU, origin at layout center, "coming from"
    azimuth_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=2 * np.pi, unit="radians"))
    ## Width parameter from ADF fit (delta_omega)
    width: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Scaling factor A from ADF fit
    scaling_factor: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## Non-reduced (raw) chi² from ADF; NaN if not filled
    ## Divide by du_count - 4, the degrees of freedom, to obtain the reduced chi²
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

    ## Cramér-Rao (lower) bound (CRB); NaN if not filled (main_DOI.py fills them, main_AOI.py does not)
    ## CRB of shower zenith angle from ADF (in radians)
    crb_zenith_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## CRB of shower azimuth angle from ADF (in radians)
    crb_azimuth_adf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## CRB of scaling factor A from ADF fit
    crb_scaling_factor: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of width parameter from ADF fit (delta_omega)
    crb_width: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower zenith from SWF (in rad)
    crb_zenith_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## CRB of shower azimuth from SWF (in rad)
    crb_azimuth_swf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## CRB of distance between the reconstructed Xsource and the origin = layout center (in meters)
    crb_r_xmax: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of emission time from SWF (in seconds)
    crb_t_s: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32))
    ## CRB of shower zenith from PWF (in radians)
    crb_zenith_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))
    ## CRB of shower azimuth from PWF (in radians)
    crb_azimuth_pwf: TTreeScalarDesc = field(default=TTreeScalarDesc(np.float32, minimum=0, maximum=np.pi, unit="radians"))

    def __post_init__(self):
        super().__post_init__()
        # An unfilled chi² or bound read 0.0, which looks like a perfect fit or
        # no uncertainty (#211); NaN says it was not computed.
        for name in self._unfilled_as_nan:
            setattr(self, name, np.nan)

    _unfilled_as_nan = ("chi2_pwf", "chi2_swf", "chi2_adf",
                        "crb_zenith_adf", "crb_azimuth_adf", "crb_scaling_factor", "crb_width",
                        "crb_zenith_swf", "crb_azimuth_swf", "crb_r_xmax", "crb_t_s",
                        "crb_zenith_pwf", "crb_azimuth_pwf")


   
