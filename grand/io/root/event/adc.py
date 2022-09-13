"""

"""

from grand.io.root.base import *
from .mother_event import MotherEventTree


@dataclass
## The class for storing ADC traces and associated values for each event
class ADCEventTree(MotherEventTree):
    """The class for storing ADC traces and associated values for each event"""

    _tree_name: str = "teventadc"

    ## Common for the whole event
    ## Event size
    _event_size: np.ndarray = np.zeros(1, np.uint32)
    ## Event in the run number
    _t3_number: np.ndarray = np.zeros(1, np.uint32)
    ## First detector unit that triggered in the event
    _first_du: np.ndarray = np.zeros(1, np.uint32)
    ## Unix time corresponding to the GPS seconds of the first triggered station
    _time_seconds: np.ndarray = np.zeros(1, np.uint32)
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = np.zeros(1, np.uint32)
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = np.zeros(1, np.uint32)
    ## Event format version of the DAQ
    _event_version: np.ndarray = np.zeros(1, np.uint32)
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = np.zeros(1, np.uint32)

    ## Specific for each Detector Unit
    ## The T3 trigger number
    _event_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Detector unit (antenna) ID
    _du_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric pressure
    _atm_pressure: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric humidity
    _atm_humidity: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in X
    _acceleration_x: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in Y
    _acceleration_y: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Acceleration of the antenna in Z
    _acceleration_z: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Battery voltage
    _battery_level: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Firmware version
    _firmware_version: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling frequency in MHz
    _adc_sampling_frequency: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC sampling resolution in bits
    _adc_sampling_resolution: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?
    _adc_input_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?
    _adc_enabled_channels: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in all channels
    _adc_samples_count_total: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 0
    _adc_samples_count_channel0: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 1
    _adc_samples_count_channel1: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 2
    _adc_samples_count_channel2: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## ADC samples callected in channel 3
    _adc_samples_count_channel3: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock tick at which the event was triggered (used to calculate the trigger time)
    _clock_tick: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Clock ticks per second
    _clock_ticks_per_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?
    _gps_offset: StdVectorList("float") = StdVectorList("float")
    ## GPS leap second
    _gps_leap_second: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS status
    _gps_status: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS alarms
    _gps_alarms: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS warnings
    _gps_warnings: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS time
    _gps_time: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Longitude
    _gps_long: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Latitude
    _gps_lat: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Altitude
    _gps_alt: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## GPS temperature
    _gps_temp: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?
    _digi_ctrl: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 0 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties0: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 1 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties1: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 2 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties2: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 3 properties - described in Channel property parameters in the manual. ToDo: Decode?
    _channel_properties3: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 0 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings0: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 1 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings1: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 2 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings2: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## Channel 3 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?
    _channel_trig_settings3: StdVectorList("vector<unsigned short>") = StdVectorList(
        "vector<unsigned short>"
    )
    ## ?? What is it? Some kind of the adc trace offset?
    _ioff: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")
    ## ADC trace 0
    _trace_0: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 1
    _trace_1: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 2
    _trace_2: StdVectorList("vector<short>") = StdVectorList("vector<short>")
    ## ADC trace 3
    _trace_3: StdVectorList("vector<short>") = StdVectorList("vector<short>")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def event_size(self):
        """Event size"""
        return self._event_size[0]

    @event_size.setter
    def event_size(self, value: np.uint32) -> None:
        self._event_size[0] = value

    @property
    def t3_number(self):
        """Event in the run number"""
        return self._t3_number[0]

    @t3_number.setter
    def t3_number(self, value: np.uint32) -> None:
        self._t3_number[0] = value

    @property
    def first_du(self):
        """First detector unit that triggered in the event"""
        return self._first_du[0]

    @first_du.setter
    def first_du(self, value: np.uint32) -> None:
        self._first_du[0] = value

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the first triggered station"""
        return self._time_seconds[0]

    @time_seconds.setter
    def time_seconds(self, value: np.uint32) -> None:
        self._time_seconds[0] = value

    @property
    def time_nanoseconds(self):
        """GPS nanoseconds corresponding to the trigger of the first triggered station"""
        return self._time_nanoseconds[0]

    @time_nanoseconds.setter
    def time_nanoseconds(self, value: np.uint32) -> None:
        self._time_nanoseconds[0] = value

    @property
    def event_type(self):
        """Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower"""
        return self._event_type[0]

    @event_type.setter
    def event_type(self, value: np.uint32) -> None:
        self._event_type[0] = value

    @property
    def event_version(self):
        """Event format version of the DAQ"""
        return self._event_version[0]

    @event_version.setter
    def event_version(self, value: np.uint32) -> None:
        self._event_version[0] = value

    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def event_id(self):
        """The T3 trigger number"""
        return self._event_id

    @event_id.setter
    def event_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._event_id.clear()
            self._event_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._event_id._vector = value
        else:
            exit(
                f"Incorrect type for event_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_id(self):
        """Detector unit (antenna) ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._du_id._vector = value
        else:
            exit(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def du_seconds(self):
        """Unix time of the trigger for this DU"""
        return self._du_seconds

    @du_seconds.setter
    def du_seconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_seconds.clear()
            self._du_seconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_seconds._vector = value
        else:
            exit(
                f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def du_nanoseconds(self):
        """Nanoseconds of the trigger for this DU"""
        return self._du_nanoseconds

    @du_nanoseconds.setter
    def du_nanoseconds(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_nanoseconds.clear()
            self._du_nanoseconds += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._du_nanoseconds._vector = value
        else:
            exit(
                f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def trigger_position(self):
        """Trigger position in the trace (trigger start = nanoseconds - 2*sample number)"""
        return self._trigger_position

    @trigger_position.setter
    def trigger_position(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_position.clear()
            self._trigger_position += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_position._vector = value
        else:
            exit(
                f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_flag(self):
        """Same as event_type, but event_type could consist of different triggered DUs"""
        return self._trigger_flag

    @trigger_flag.setter
    def trigger_flag(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_flag.clear()
            self._trigger_flag += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_flag._vector = value
        else:
            exit(
                f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_temperature(self):
        """Atmospheric temperature (read via I2C)"""
        return self._atm_temperature

    @atm_temperature.setter
    def atm_temperature(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_temperature.clear()
            self._atm_temperature += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_temperature._vector = value
        else:
            exit(
                f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_pressure(self):
        """Atmospheric pressure"""
        return self._atm_pressure

    @atm_pressure.setter
    def atm_pressure(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_pressure.clear()
            self._atm_pressure += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_pressure._vector = value
        else:
            exit(
                f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def atm_humidity(self):
        """Atmospheric humidity"""
        return self._atm_humidity

    @atm_humidity.setter
    def atm_humidity(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._atm_humidity.clear()
            self._atm_humidity += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._atm_humidity._vector = value
        else:
            exit(
                f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_x(self):
        """Acceleration of the antenna in X"""
        return self._acceleration_x

    @acceleration_x.setter
    def acceleration_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_x.clear()
            self._acceleration_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_x._vector = value
        else:
            exit(
                f"Incorrect type for acceleration_x {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_y(self):
        """Acceleration of the antenna in Y"""
        return self._acceleration_y

    @acceleration_y.setter
    def acceleration_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_y.clear()
            self._acceleration_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_y._vector = value
        else:
            exit(
                f"Incorrect type for acceleration_y {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def acceleration_z(self):
        """Acceleration of the antenna in Z"""
        return self._acceleration_z

    @acceleration_z.setter
    def acceleration_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._acceleration_z.clear()
            self._acceleration_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._acceleration_z._vector = value
        else:
            exit(
                f"Incorrect type for acceleration_z {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def battery_level(self):
        """Battery voltage"""
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._battery_level.clear()
            self._battery_level += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._battery_level._vector = value
        else:
            exit(
                f"Incorrect type for battery_level {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def firmware_version(self):
        """Firmware version"""
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._firmware_version.clear()
            self._firmware_version += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._firmware_version._vector = value
        else:
            exit(
                f"Incorrect type for firmware_version {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_sampling_frequency(self):
        """ADC sampling frequency in MHz"""
        return self._adc_sampling_frequency

    @adc_sampling_frequency.setter
    def adc_sampling_frequency(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_frequency.clear()
            self._adc_sampling_frequency += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_frequency._vector = value
        else:
            exit(
                f"Incorrect type for adc_sampling_frequency {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_sampling_resolution(self):
        """ADC sampling resolution in bits"""
        return self._adc_sampling_resolution

    @adc_sampling_resolution.setter
    def adc_sampling_resolution(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_sampling_resolution.clear()
            self._adc_sampling_resolution += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_sampling_resolution._vector = value
        else:
            exit(
                f"Incorrect type for adc_sampling_resolution {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_input_channels(self):
        """ADC input channels - > 16 BIT WORD (4*4 BITS) LOWEST IS CHANNEL 1, HIGHEST CHANNEL 4. FOR EACH CHANNEL IN THE EVENT WE HAVE: 0: ADC1, 1: ADC2, 2:ADC3, 3:ADC4 4:FILTERED ADC1, 5:FILTERED ADC 2, 6:FILTERED ADC3, 7:FILTERED ADC4. ToDo: decode this?"""
        return self._adc_input_channels

    @adc_input_channels.setter
    def adc_input_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_input_channels.clear()
            self._adc_input_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_input_channels._vector = value
        else:
            exit(
                f"Incorrect type for adc_input_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_enabled_channels(self):
        """ADC enabled channels - LOWEST 4 BITS STATE WHICH CHANNEL IS READ OUT ToDo: Decode this?"""
        return self._adc_enabled_channels

    @adc_enabled_channels.setter
    def adc_enabled_channels(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_enabled_channels.clear()
            self._adc_enabled_channels += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_enabled_channels._vector = value
        else:
            exit(
                f"Incorrect type for adc_enabled_channels {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_total(self):
        """ADC samples callected in all channels"""
        return self._adc_samples_count_total

    @adc_samples_count_total.setter
    def adc_samples_count_total(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_total.clear()
            self._adc_samples_count_total += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_total._vector = value
        else:
            exit(
                f"Incorrect type for adc_samples_count_total {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel0(self):
        """ADC samples callected in channel 0"""
        return self._adc_samples_count_channel0

    @adc_samples_count_channel0.setter
    def adc_samples_count_channel0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel0.clear()
            self._adc_samples_count_channel0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel0._vector = value
        else:
            exit(
                f"Incorrect type for adc_samples_count_channel0 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel1(self):
        """ADC samples callected in channel 1"""
        return self._adc_samples_count_channel1

    @adc_samples_count_channel1.setter
    def adc_samples_count_channel1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel1.clear()
            self._adc_samples_count_channel1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel1._vector = value
        else:
            exit(
                f"Incorrect type for adc_samples_count_channel1 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel2(self):
        """ADC samples callected in channel 2"""
        return self._adc_samples_count_channel2

    @adc_samples_count_channel2.setter
    def adc_samples_count_channel2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel2.clear()
            self._adc_samples_count_channel2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel2._vector = value
        else:
            exit(
                f"Incorrect type for adc_samples_count_channel2 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def adc_samples_count_channel3(self):
        """ADC samples callected in channel 3"""
        return self._adc_samples_count_channel3

    @adc_samples_count_channel3.setter
    def adc_samples_count_channel3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._adc_samples_count_channel3.clear()
            self._adc_samples_count_channel3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._adc_samples_count_channel3._vector = value
        else:
            exit(
                f"Incorrect type for adc_samples_count_channel3 {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_pattern(self):
        """Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?"""
        return self._trigger_pattern

    @trigger_pattern.setter
    def trigger_pattern(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_pattern.clear()
            self._trigger_pattern += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_pattern._vector = value
        else:
            exit(
                f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trigger_rate(self):
        """Trigger rate - the number of triggers recorded in the second preceding the event"""
        return self._trigger_rate

    @trigger_rate.setter
    def trigger_rate(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trigger_rate.clear()
            self._trigger_rate += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._trigger_rate._vector = value
        else:
            exit(
                f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def clock_tick(self):
        """Clock tick at which the event was triggered (used to calculate the trigger time)"""
        return self._clock_tick

    @clock_tick.setter
    def clock_tick(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_tick.clear()
            self._clock_tick += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_tick._vector = value
        else:
            exit(
                f"Incorrect type for clock_tick {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def clock_ticks_per_second(self):
        """Clock ticks per second"""
        return self._clock_ticks_per_second

    @clock_ticks_per_second.setter
    def clock_ticks_per_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._clock_ticks_per_second.clear()
            self._clock_ticks_per_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._clock_ticks_per_second._vector = value
        else:
            exit(
                f"Incorrect type for clock_ticks_per_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_offset(self):
        """GPS offset - offset between the PPS and the real second (in GPS). ToDo: is it already included in the time calculations?"""
        return self._gps_offset

    @gps_offset.setter
    def gps_offset(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_offset.clear()
            self._gps_offset += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._gps_offset._vector = value
        else:
            exit(
                f"Incorrect type for gps_offset {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def gps_leap_second(self):
        """GPS leap second"""
        return self._gps_leap_second

    @gps_leap_second.setter
    def gps_leap_second(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_leap_second.clear()
            self._gps_leap_second += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_leap_second._vector = value
        else:
            exit(
                f"Incorrect type for gps_leap_second {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_status(self):
        """GPS status"""
        return self._gps_status

    @gps_status.setter
    def gps_status(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_status.clear()
            self._gps_status += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_status._vector = value
        else:
            exit(
                f"Incorrect type for gps_status {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_alarms(self):
        """GPS alarms"""
        return self._gps_alarms

    @gps_alarms.setter
    def gps_alarms(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alarms.clear()
            self._gps_alarms += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alarms._vector = value
        else:
            exit(
                f"Incorrect type for gps_alarms {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_warnings(self):
        """GPS warnings"""
        return self._gps_warnings

    @gps_warnings.setter
    def gps_warnings(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_warnings.clear()
            self._gps_warnings += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_warnings._vector = value
        else:
            exit(
                f"Incorrect type for gps_warnings {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_time(self):
        """GPS time"""
        return self._gps_time

    @gps_time.setter
    def gps_time(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_time.clear()
            self._gps_time += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned int")):
            self._gps_time._vector = value
        else:
            exit(
                f"Incorrect type for gps_time {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required."
            )

    @property
    def gps_long(self):
        """Longitude"""
        return self._gps_long

    @gps_long.setter
    def gps_long(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_long.clear()
            self._gps_long += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_long._vector = value
        else:
            exit(
                f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_lat(self):
        """Latitude"""
        return self._gps_lat

    @gps_lat.setter
    def gps_lat(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_lat.clear()
            self._gps_lat += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_lat._vector = value
        else:
            exit(
                f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_alt(self):
        """Altitude"""
        return self._gps_alt

    @gps_alt.setter
    def gps_alt(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_alt.clear()
            self._gps_alt += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_alt._vector = value
        else:
            exit(
                f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def gps_temp(self):
        """GPS temperature"""
        return self._gps_temp

    @gps_temp.setter
    def gps_temp(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._gps_temp.clear()
            self._gps_temp += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._gps_temp._vector = value
        else:
            exit(
                f"Incorrect type for gps_temp {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def digi_ctrl(self):
        """Control parameters - the list of general parameters that can set the mode of operation, select trigger sources and preset the common coincidence read out time window (Digitizer mode parameters in the manual). ToDo: Decode?"""
        return self._digi_ctrl

    @digi_ctrl.setter
    def digi_ctrl(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_ctrl.clear()
            self._digi_ctrl += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_ctrl._vector = value
        else:
            exit(
                f"Incorrect type for digi_ctrl {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def digi_prepost_trig_windows(self):
        """Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?"""
        return self._digi_prepost_trig_windows

    @digi_prepost_trig_windows.setter
    def digi_prepost_trig_windows(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._digi_prepost_trig_windows.clear()
            self._digi_prepost_trig_windows += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._digi_prepost_trig_windows._vector = value
        else:
            exit(
                f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties0(self):
        """Channel 0 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties0

    @channel_properties0.setter
    def channel_properties0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties0.clear()
            self._channel_properties0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties0._vector = value
        else:
            exit(
                f"Incorrect type for channel_properties0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties1(self):
        """Channel 1 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties1

    @channel_properties1.setter
    def channel_properties1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties1.clear()
            self._channel_properties1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties1._vector = value
        else:
            exit(
                f"Incorrect type for channel_properties1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties2(self):
        """Channel 2 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties2

    @channel_properties2.setter
    def channel_properties2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties2.clear()
            self._channel_properties2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties2._vector = value
        else:
            exit(
                f"Incorrect type for channel_properties2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_properties3(self):
        """Channel 3 properties - described in Channel property parameters in the manual. ToDo: Decode?"""
        return self._channel_properties3

    @channel_properties3.setter
    def channel_properties3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_properties3.clear()
            self._channel_properties3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_properties3._vector = value
        else:
            exit(
                f"Incorrect type for channel_properties3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings0(self):
        """Channel 0 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings0

    @channel_trig_settings0.setter
    def channel_trig_settings0(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings0.clear()
            self._channel_trig_settings0 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings0._vector = value
        else:
            exit(
                f"Incorrect type for channel_trig_settings0 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings1(self):
        """Channel 1 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings1

    @channel_trig_settings1.setter
    def channel_trig_settings1(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings1.clear()
            self._channel_trig_settings1 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings1._vector = value
        else:
            exit(
                f"Incorrect type for channel_trig_settings1 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings2(self):
        """Channel 2 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings2

    @channel_trig_settings2.setter
    def channel_trig_settings2(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings2.clear()
            self._channel_trig_settings2 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings2._vector = value
        else:
            exit(
                f"Incorrect type for channel_trig_settings2 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def channel_trig_settings3(self):
        """Channel 3 trigger settings - described in Channel trigger parameters in the manual. ToDo: Decode?"""
        return self._channel_trig_settings3

    @channel_trig_settings3.setter
    def channel_trig_settings3(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._channel_trig_settings3.clear()
            self._channel_trig_settings3 += value
        # A vector was given
        elif isinstance(value, ROOT.vector("vector<unsigned short>")):
            self._channel_trig_settings3._vector = value
        else:
            exit(
                f"Incorrect type for channel_trig_settings3 {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required."
            )

    @property
    def ioff(self):
        """?? What is it? Some kind of the adc trace offset?"""
        return self._ioff

    @ioff.setter
    def ioff(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._ioff.clear()
            self._ioff += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._ioff._vector._vector = value
        else:
            exit(
                f"Incorrect type for ioff {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )

    @property
    def trace_0(self):
        """ADC trace 0"""
        return self._trace_0

    @trace_0.setter
    def trace_0(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_0.clear()
            self._trace_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            # With vectors, I think the address is assigned, so in principle the below is needed only on the first setting of the branch
            self._trace_0._vector = value
        else:
            exit(
                f"Incorrect type for trace_0 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
            )

    @property
    def trace_1(self):
        """ADC trace 1"""
        return self._trace_1

    @trace_1.setter
    def trace_1(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_1.clear()
            self._trace_1 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_1._vector = value
        else:
            exit(
                f"Incorrect type for trace_1 {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )

    @property
    def trace_2(self):
        """ADC trace 2"""
        return self._trace_2

    @trace_2.setter
    def trace_2(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_2.clear()
            self._trace_2 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_2._vector = value
        else:
            exit(
                f"Incorrect type for trace_2 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
            )

    @property
    def trace_3(self):
        """ADC trace 3"""
        return self._trace_3

    @trace_3.setter
    def trace_3(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_3.clear()
            self._trace_3 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<short>")):
            self._trace_3._vector = value
        else:
            exit(
                f"Incorrect type for trace_3 {type(value)}. Either a list, an array or a ROOT.vector of vector<short> required."
            )
