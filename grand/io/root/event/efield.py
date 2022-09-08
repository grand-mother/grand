'''

'''

from grand.io.root.base import *
from .mother_event import MotherEventTree


@dataclass
## The class for storing Efield traces and associated values for each event
class EfieldEventTree(MotherEventTree):
    """The class for storing Efield traces and associated values for each event"""
    _tree_name: str = "teventefield"

    # _du_id: StdVectorList("int") = StdVectorList("int")
    # _event_size: np.ndarray = np.zeros(1, np.uint32)
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")

    ## Common for the whole event
    ## Unix time corresponding to the GPS seconds of the trigger
    _time_seconds: np.ndarray = np.zeros(1, np.uint32)
    ## GPS nanoseconds corresponding to the trigger of the first triggered station
    _time_nanoseconds: np.ndarray = np.zeros(1, np.uint32)
    ## Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    _event_type: np.ndarray = np.zeros(1, np.uint32)
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = np.zeros(1, np.uint32)


    ## Specific for each Detector Unit
    ## Detector unit (antenna) ID
    _du_id: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Unix time of the trigger for this DU
    _du_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the trigger for this DU
    _du_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Unix time of the start of the trace for this DU
    # _du_t0_seconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Nanoseconds of the start of the trace for this DU
    # _du_t0_nanoseconds: StdVectorList("unsigned int") = StdVectorList("unsigned int")
    ## Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    _trigger_position: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Same as event_type, but event_type could consist of different triggered DUs
    _trigger_flag: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Atmospheric temperature (read via I2C)
    _atm_temperature: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric pressure
    _atm_pressure: StdVectorList("float") = StdVectorList("float")
    ## Atmospheric humidity
    _atm_humidity: StdVectorList("float") = StdVectorList("float")
    ## Trigger pattern - which of the trigger sources (more than one may be present) fired to actually the trigger the digitizer - explained in the docs. ToDo: Decode this?
    _trigger_pattern: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Trigger rate - the number of triggers recorded in the second preceding the event
    _trigger_rate: StdVectorList("unsigned short") = StdVectorList("unsigned short")
    ## Longitude
    _gps_long: StdVectorList("float") = StdVectorList("float")
    ## Latitude
    _gps_lat: StdVectorList("float") = StdVectorList("float")
    ## Altitude
    _gps_alt: StdVectorList("float") = StdVectorList("float")
    ## X position in site's referential
    _pos_x: StdVectorList("float") = StdVectorList("float")
    ## Y position in site's referential
    _pos_y: StdVectorList("float") = StdVectorList("float")
    ## Z position in site's referential
    _pos_z: StdVectorList("float") = StdVectorList("float")
    ## Window parameters - describe Pre Coincidence, Coincidence and Post Coincidence readout windows (Digitizer window parameters in the manual). ToDo: Decode?
    _digi_prepost_trig_windows: StdVectorList("vector<unsigned short>") = StdVectorList("vector<unsigned short>")

    ## Efield trace in X direction
    _trace_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Efield trace in Y direction
    _trace_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Efield trace in Z direction
    _trace_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in X direction
    _fft_mag_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in Y direction
    _fft_mag_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT magnitude in Z direction
    _fft_mag_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in X direction
    _fft_phase_x: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in Y direction
    _fft_phase_y: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## FFT phase in Z direction
    _fft_phase_z: StdVectorList("vector<float>") = StdVectorList("vector<float>")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName()=="":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def time_seconds(self):
        """Unix time corresponding to the GPS seconds of the trigger"""
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
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

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
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

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
            exit(f"Incorrect type for du_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

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
            exit(f"Incorrect type for du_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

    # @property
    # def du_t0_seconds(self):
    #     return self._du_t0_seconds
    #
    # @du_t0_seconds.setter
    # def du_t0_seconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_seconds.clear()
    #         self._du_t0_seconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_seconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_seconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")
    #
    # @property
    # def du_t0_nanoseconds(self):
    #     return self._du_t0_nanoseconds
    #
    # @du_t0_nanoseconds.setter
    # def du_t0_nanoseconds(self, value) -> None:
    #     # A list of strings was given
    #     if isinstance(value, list) or isinstance(value, np.ndarray):
    #         # Clear the vector before setting
    #         self._du_t0_nanoseconds.clear()
    #         self._du_t0_nanoseconds += value
    #     # A vector was given
    #     elif isinstance(value, ROOT.vector("unsigned int")):
    #         self._du_t0_nanoseconds._vector = value
    #     else:
    #         exit(f"Incorrect type for du_t0_nanoseconds {type(value)}. Either a list, an array or a ROOT.vector of unsigned ints required.")

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
            exit(f"Incorrect type for trigger_position {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

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
            exit(f"Incorrect type for trigger_flag {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._atm_temperature._vector = value
        else:
            exit(f"Incorrect type for atm_temperature {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._atm_pressure._vector = value
        else:
            exit(f"Incorrect type for atm_pressure {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._atm_humidity._vector = value
        else:
            exit(f"Incorrect type for atm_humidity {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
            exit(f"Incorrect type for trigger_pattern {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

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
            exit(f"Incorrect type for trigger_rate {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._gps_long._vector = value
        else:
            exit(f"Incorrect type for gps_long {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._gps_lat._vector = value
        else:
            exit(f"Incorrect type for gps_lat {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
        elif isinstance(value, ROOT.vector("float")):
            self._gps_alt._vector = value
        else:
            exit(f"Incorrect type for gps_alt {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_x(self):
        """X position in site's referential"""
        return self._pos_x

    @pos_x.setter
    def pos_x(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_x.clear()
            self._pos_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_x._vector = value
        else:
            exit(f"Incorrect type for pos_x {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_y(self):
        """Y position in site's referential"""
        return self._pos_y

    @pos_y.setter
    def pos_y(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_y.clear()
            self._pos_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_y._vector = value
        else:
            exit(f"Incorrect type for pos_y {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

    @property
    def pos_z(self):
        """Z position in site's referential"""
        return self._pos_z

    @pos_z.setter
    def pos_z(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._pos_z.clear()
            self._pos_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._pos_z._vector = value
        else:
            exit(f"Incorrect type for pos_z {type(value)}. Either a list, an array or a ROOT.vector of floats required.")

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
            exit(f"Incorrect type for digi_prepost_trig_windows {type(value)}. Either a list, an array or a ROOT.vector of vector<unsigned short>s required.")

    @property
    def trace_x(self):
        """Efield trace in X direction"""
        return self._trace_x

    @trace_x.setter
    def trace_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_x.clear()
            self._trace_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_x._vector = value
        else:
            exit(f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required.")

    @property
    def trace_y(self):
        """Efield trace in Y direction"""
        return self._trace_y

    @trace_y.setter
    def trace_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_y.clear()
            self._trace_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_y._vector = value
        else:
            exit(f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def trace_z(self):
        """Efield trace in Z direction"""
        return self._trace_z

    @trace_z.setter
    def trace_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._trace_z.clear()
            self._trace_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_z._vector = value
        else:
            exit(f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_x(self):
        """FFT magnitude in X direction"""
        return self._fft_mag_x

    @fft_mag_x.setter
    def fft_mag_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_x.clear()
            self._fft_mag_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_x._vector = value
        else:
            exit(f"Incorrect type for fft_mag_x {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_y(self):
        """FFT magnitude in Y direction"""
        return self._fft_mag_y

    @fft_mag_y.setter
    def fft_mag_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_y.clear()
            self._fft_mag_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_y._vector = value
        else:
            exit(f"Incorrect type for fft_mag_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_mag_z(self):
        """FFT magnitude in Z direction"""
        return self._fft_mag_z

    @fft_mag_z.setter
    def fft_mag_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_mag_z.clear()
            self._fft_mag_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_mag_z._vector = value
        else:
            exit(f"Incorrect type for fft_mag_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_x(self):
        """FFT phase in X direction"""
        return self._fft_phase_x

    @fft_phase_x.setter
    def fft_phase_x(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_x.clear()
            self._fft_phase_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_x._vector = value
        else:
            exit(f"Incorrect type for fft_phase_x {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_y(self):
        """FFT phase in Y direction"""
        return self._fft_phase_y

    @fft_phase_y.setter
    def fft_phase_y(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_y.clear()
            self._fft_phase_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_y._vector = value
        else:
            exit(f"Incorrect type for fft_phase_y {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def fft_phase_z(self):
        """FFT phase in Z direction"""
        return self._fft_phase_z

    @fft_phase_z.setter
    def fft_phase_z(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._fft_phase_z.clear()
            self._fft_phase_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._fft_phase_z._vector = value
        else:
            exit(f"Incorrect type for fft_phase_z {type(value)}. Either a list, an array or a ROOT.vector of float required.")



@dataclass
## The class for storing Efield simulation-only data common for each event
class EfieldEventSimdataTree(MotherEventTree):
    """The class for storing Efield simulation-only data common for each event"""
    _tree_name: str = "teventefieldsimdata"

    _du_id: StdVectorList("int") = StdVectorList("int")  # Detector ID
    _t_0: StdVectorList("float") = StdVectorList("float")  # Time window t0
    _p2p: StdVectorList("float") = StdVectorList("float")  # peak 2 peak amplitudes (x,y,z,modulus)

    # _event_size: np.ndarray = np.zeros(1, np.uint32)
    # _start_time: StdVectorList("double") = StdVectorList("double")
    # _rel_peak_time: StdVectorList("float") = StdVectorList("float")
    # _det_time: StdVectorList("double") = StdVectorList("double")
    # _e_det_time: StdVectorList("double") = StdVectorList("double")
    # _isTriggered: StdVectorList("bool") = StdVectorList("bool")
    # _sampling_speed: StdVectorList("float") = StdVectorList("float")


    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def du_id(self):
        """Detector ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_id._vector = value
        else:
            exit(f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def t_0(self):
        """Time window t0"""
        return self._t_0

    @t_0.setter
    def t_0(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._t_0.clear()
            self._t_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._t_0._vector = value
        else:
            exit(f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required.")

    @property
    def p2p(self):
        """Peak 2 peak amplitudes (x,y,z,modulus)"""
        return self._p2p

    @p2p.setter
    def p2p(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._p2p._vector = value
        else:
            exit(f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required.")


