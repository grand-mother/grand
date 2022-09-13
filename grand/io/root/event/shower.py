"""

"""

from grand.io.root.base import *
from .mother_event import MotherEventTree


@dataclass
## The class for storing shower data common for each event
class ShowerEventTree(MotherEventTree):
    """The class for storing reconstructed shower data common for each event"""

    _tree_name: str = "teventshower"

    _shower_type: StdString = StdString(
        ""
    )  # shower primary type: If single particle, particle type. If not...tau decay,etc. TODO: Standarize
    _shower_energy: np.ndarray = np.zeros(
        1, np.float32
    )  # shower energy (GeV)  Check unit conventions.
    _shower_azimuth: np.ndarray = np.zeros(
        1, np.float32
    )  # shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem
    _shower_zenith: np.ndarray = np.zeros(
        1, np.float32
    )  # shower zenith  TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth
    _shower_core_pos: np.ndarray = np.zeros(
        4, np.float32
    )  # shower core position TODO: Coordinates in geoid?. Undefined for neutrinos.
    _atmos_model: StdString = StdString("")  # Atmospheric model name TODO:standarize
    _atmos_model_param: np.ndarray = np.zeros(
        3, np.float32
    )  # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    _magnetic_field: np.ndarray = np.zeros(
        3, np.float32
    )  # Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense.
    _date: StdString = StdString(
        ""
    )  # Event Date and time. TODO:standarize (date format, time format)
    _ground_alt: np.ndarray = np.zeros(1, np.float32)  # Ground Altitude (m)
    _xmax_grams: np.ndarray = np.zeros(
        1, np.float32
    )  # shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_pos_shc: np.ndarray = np.zeros(
        3, np.float64
    )  # shower Xmax position in shower coordinates
    _xmax_alt: np.ndarray = np.zeros(
        1, np.float64
    )  # altitude of Xmax  (m, in the shower simulation earth. Its important for the index of refraction )
    _gh_fit_param: np.ndarray = np.zeros(
        3, np.float32
    )  # X0,Xmax,Lambda (g/cm2) (3 parameter GH function fit to the longitudinal development of all particles)
    _core_time: np.ndarray = np.zeros(
        1, np.float64
    )  # ToDo: Check; time when the shower was at the core position - defined in Charles, but not in Zhaires/Coreas?

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def shower_type(self):
        """Shower primary type: If single particle, particle type. If not...tau decay,etc. TODO: Standarize"""
        return str(self._shower_type)

    @shower_type.setter
    def shower_type(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._shower_type.string.assign(value)

    @property
    def shower_energy(self):
        """Shower energy (GeV). ToDo: Check unit conventions."""
        return self._shower_energy[0]

    @shower_energy.setter
    def shower_energy(self, value):
        self._shower_energy[0] = value

    @property
    def shower_azimuth(self):
        """Shower azimuth. TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neutrino convention is problematic for round earth. Also, geoid vs sphere problem"""
        return self._shower_azimuth[0]

    @shower_azimuth.setter
    def shower_azimuth(self, value):
        self._shower_azimuth[0] = value

    @property
    def shower_zenith(self):
        """Shower zenith. TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neutrino convention is problematic for round earth"""
        return self._shower_zenith[0]

    @shower_zenith.setter
    def shower_zenith(self, value):
        self._shower_zenith[0] = value

    @property
    def shower_core_pos(self):
        """Shower core position TODO: Coordinates in geoid?. Undefined for neutrinos."""
        return np.array(self._shower_core_pos)

    @shower_core_pos.setter
    def shower_core_pos(self, value):
        self._shower_core_pos = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("shower_core_pos", self._shower_core_pos)

    @property
    def atmos_model(self):
        """Atmospheric model name. TODO: standarize"""
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._atmos_model.string.assign(value)

    @property
    def atmos_model_param(self):
        """Atmospheric model parameters. TODO: Think about this. Different models and softwares can have different parameters"""
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)

    @property
    def magnetic_field(self):
        """Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense."""
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)

    @property
    def date(self):
        """Event Date and time. TODO: standarize (date format, time format)"""
        return str(self._date)

    @date.setter
    def date(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._date.string.assign(value)

    @property
    def ground_alt(self):
        """Ground Altitude (m)"""
        return self._ground_alt[0]

    @ground_alt.setter
    def ground_alt(self, value):
        self._ground_alt[0] = value

    @property
    def xmax_grams(self):
        """Shower Xmax depth (g/cm2 along the shower axis)."""
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value

    @property
    def xmax_pos_shc(self):
        """Shower Xmax position in shower coordinates."""
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float64)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)

    @property
    def xmax_alt(self):
        """Altitude of Xmax (m, in the shower simulation earth. It's important for the index of refraction)."""
        return self._xmax_alt[0]

    @xmax_alt.setter
    def xmax_alt(self, value):
        self._xmax_alt[0] = value

    @property
    def gh_fit_param(self):
        """X0,Xmax,Lambda (g/cm2) (3 parameter GH function fit to the longitudinal development of all particles)."""
        return np.array(self._gh_fit_param)

    @gh_fit_param.setter
    def gh_fit_param(self, value):
        self._gh_fit_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("gh_fit_param", self._gh_fit_param)

    @property
    def core_time(self):
        """ToDo: Check; time when the shower was at the core position - defined in Charles, but not in Zhaires/Coreas?"""
        return self._core_time[0]

    @core_time.setter
    def core_time(self, value):
        self._core_time[0] = value


@dataclass
## The class for storing a shower simulation-only data for each event
class ShowerEventSimdataTree(MotherEventTree):
    """The class for storing a shower simulation-only data for each event"""

    _tree_name: str = "teventshowersimdata"

    ## Event name
    _event_name: StdString = StdString("")

    ## Event Date and time. TODO:standarize (date format, time format)
    _date: StdString = StdString("")
    ## Random seed
    _rnd_seed: np.ndarray = np.zeros(1, np.float64)
    ## Energy in neutrinos generated in the shower (GeV). Usefull for invisible energy
    _energy_in_neutrinos: np.ndarray = np.zeros(1, np.float32)
    # _prim_energy: np.ndarray = np.zeros(1, np.float32)  # primary energy (GeV) TODO: Support multiple primaries. Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector
    ## Primary energy (GeV) TODO: Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector
    _prim_energy: StdVectorList("float") = StdVectorList("float")
    ## Shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem
    _shower_azimuth: np.ndarray = np.zeros(1, np.float32)
    ## Shower zenith  TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth
    _shower_zenith: np.ndarray = np.zeros(1, np.float32)
    # _prim_type: StdVectorList("string") = StdVectorList("string")  # primary particle type TODO: Support multiple primaries. standarize (PDG?)
    ## Primary particle type TODO: standarize (PDG?)
    _prim_type: StdVectorList("string") = StdVectorList("string")
    # _prim_injpoint_shc: np.ndarray = np.zeros(4, np.float32)  # primary injection point in Shower coordinates TODO: Support multiple primaries
    ## Primary injection point in Shower coordinates
    _prim_injpoint_shc: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    # _prim_inj_alt_shc: np.ndarray = np.zeros(1, np.float32)  # primary injection altitude in Shower Coordinates TODO: Support multiple primaries
    ## Primary injection altitude in Shower Coordinates
    _prim_inj_alt_shc: StdVectorList("float") = StdVectorList("float")
    # _prim_inj_dir_shc: np.ndarray = np.zeros(3, np.float32)  # primary injection direction in Shower Coordinates  TODO: Support multiple primaries
    ## primary injection direction in Shower Coordinates
    _prim_inj_dir_shc: StdVectorList("vector<float>") = StdVectorList("vector<float>")
    ## Atmospheric model name TODO:standarize
    _atmos_model: StdString = StdString("")
    # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    _atmos_model_param: np.ndarray = np.zeros(3, np.float32)
    # Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense.
    _magnetic_field: np.ndarray = np.zeros(3, np.float32)
    ## Shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_grams: np.ndarray = np.zeros(1, np.float32)
    ## Shower Xmax position in shower coordinates
    _xmax_pos_shc: np.ndarray = np.zeros(3, np.float64)
    ## Distance of Xmax  [m]
    _xmax_distance: np.ndarray = np.zeros(1, np.float64)
    ## Altitude of Xmax  (m, in the shower simulation earth. Its important for the index of refraction )
    _xmax_alt: np.ndarray = np.zeros(1, np.float64)
    _hadronic_model: StdString = StdString(
        ""
    )  # high energy hadronic model (and version) used TODO: standarize
    _low_energy_model: StdString = StdString(
        ""
    )  # high energy model (and version) used TODO: standarize
    _cpu_time: np.ndarray = np.zeros(
        1, np.float32
    )  # Time it took for the simulation. In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def event_name(self):
        """Event name"""
        return str(self._event_name)

    @event_name.setter
    def event_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for event_name {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._event_name.string.assign(value)

    @property
    def date(self):
        """Event Date and time. TODO:standarize (date format, time format)"""
        return str(self._date)

    @date.setter
    def date(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for date {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._date.string.assign(value)

    @property
    def rnd_seed(self):
        """Random seed"""
        return self._rnd_seed[0]

    @rnd_seed.setter
    def rnd_seed(self, value):
        self._rnd_seed[0] = value

    @property
    def energy_in_neutrinos(self):
        """Energy in neutrinos generated in the shower (GeV). Usefull for invisible energy"""
        return self._energy_in_neutrinos[0]

    @energy_in_neutrinos.setter
    def energy_in_neutrinos(self, value):
        self._energy_in_neutrinos[0] = value

    @property
    def prim_energy(self):
        """Primary energy (GeV) TODO: Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector"""
        return self._prim_energy

    @prim_energy.setter
    def prim_energy(self, value):
        # A list of strings was given
        if isinstance(value, list):
            # Clear the vector before setting
            self._prim_energy.clear()
            self._prim_energy += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._prim_energy._vector = value
        else:
            exit(
                f"Incorrect type for prim_energy {type(value)}. Either a list or a ROOT.vector of floats required."
            )

    @property
    def shower_azimuth(self):
        """Shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem"""
        return self._shower_azimuth[0]

    @shower_azimuth.setter
    def shower_azimuth(self, value):
        self._shower_azimuth[0] = value

    @property
    def shower_zenith(self):
        """Shower zenith TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth"""
        return self._shower_zenith[0]

    @shower_zenith.setter
    def shower_zenith(self, value):
        self._shower_zenith[0] = value

    @property
    def prim_type(self):
        """Primary particle type TODO: standarize (PDG?)"""
        return self._prim_type

    @prim_type.setter
    def prim_type(self, value):
        # A list of strings was given
        if isinstance(value, list):
            # Clear the vector before setting
            self._prim_type.clear()
            self._prim_type += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<string>")):
            self._prim_type._vector = value
        else:
            exit(
                f"Incorrect type for prim_type {type(value)}. Either a list or a ROOT.vector of strings required."
            )

    @property
    def prim_injpoint_shc(self):
        """Primary injection point in Shower coordinates"""
        return np.array(self._prim_injpoint_shc)

    @prim_injpoint_shc.setter
    def prim_injpoint_shc(self, value):
        self._prim_injpoint_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("prim_injpoint_shc", self._prim_injpoint_shc)

    @property
    def prim_inj_alt_shc(self):
        """Primary injection altitude in Shower Coordinates"""
        return self._prim_inj_alt_shc

    @prim_inj_alt_shc.setter
    def prim_inj_alt_shc(self, value):
        # A list was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._prim_inj_alt_shc.clear()
            self._prim_inj_alt_shc += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._prim_inj_alt_shc._vector = value
        else:
            exit(
                f"Incorrect type for prim_inj_alt_shc {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def prim_inj_dir_shc(self):
        """primary injection direction in Shower Coordinates"""
        return np.array(self._prim_inj_dir_shc)

    @prim_inj_dir_shc.setter
    def prim_inj_dir_shc(self, value):
        self._prim_inj_dir_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("prim_inj_dir_shc", self._prim_inj_dir_shc)

    @property
    def atmos_model(self):
        """Atmospheric model name TODO:standarize"""
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._atmos_model.string.assign(value)

    @property
    def atmos_model_param(self):
        """Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters"""
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)

    @property
    def magnetic_field(self):
        """Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense."""
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)

    @property
    def xmax_grams(self):
        """Shower Xmax depth (g/cm2 along the shower axis)"""
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value

    @property
    def xmax_pos_shc(self):
        """Shower Xmax position in shower coordinates"""
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)

    @property
    def xmax_distance(self):
        """Distance of Xmax [m]"""
        return self._xmax_distance[0]

    @xmax_distance.setter
    def xmax_distance(self, value):
        self._xmax_distance[0] = value

    @property
    def xmax_alt(self):
        """Altitude of Xmax (m, in the shower simulation earth. Its important for the index of refraction )"""
        return self._xmax_alt[0]

    @xmax_alt.setter
    def xmax_alt(self, value):
        self._xmax_alt[0] = value

    @property
    def hadronic_model(self):
        """High energy hadronic model (and version) used TODO: standarize"""
        return str(self._hadronic_model)

    @hadronic_model.setter
    def hadronic_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._hadronic_model.string.assign(value)

    @property
    def low_energy_model(self):
        """High energy model (and version) used TODO: standarize"""
        return str(self._low_energy_model)

    @low_energy_model.setter
    def low_energy_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._low_energy_model.string.assign(value)

    @property
    def cpu_time(self):
        """Time it took for the simulation. In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation"""
        return np.array(self._cpu_time)

    @cpu_time.setter
    def cpu_time(self, value):
        self._cpu_time = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("cpu_time", self._cpu_time)


@dataclass
## The class for storing shower data for each event specific to ZHAireS only
class ShowerEventZHAireSTree(MotherEventTree):
    """The class for storing shower data for each event specific to ZHAireS only"""

    _tree_name: str = "teventshowerzhaires"

    # ToDo: we need explanations of these parameters

    _relative_thining: StdString = StdString("")
    _weight_factor: np.ndarray = np.zeros(1, np.float64)
    _gamma_energy_cut: StdString = StdString("")
    _electron_energy_cut: StdString = StdString("")
    _muon_energy_cut: StdString = StdString("")
    _meson_energy_cut: StdString = StdString("")
    _nucleon_energy_cut: StdString = StdString("")
    _other_parameters: StdString = StdString("")

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def relative_thining(self):
        """Relative thinning energy"""
        return str(self._relative_thining)

    @relative_thining.setter
    def relative_thining(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for relative_thining {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._relative_thining.string.assign(value)

    @property
    def weight_factor(self):
        """Weight factor"""
        return self._weight_factor[0]

    @weight_factor.setter
    def weight_factor(self, value: np.float64) -> None:
        self._weight_factor[0] = value

    @property
    def gamma_energy_cut(self):
        """Low energy cut for gammas(GeV)"""
        return str(self._gamma_energy_cut)

    @gamma_energy_cut.setter
    def gamma_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for gamma_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._gamma_energy_cut.string.assign(value)

    @property
    def electron_energy_cut(self):
        """Low energy cut for electrons (GeV)"""
        return str(self._electron_energy_cut)

    @electron_energy_cut.setter
    def electron_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for electron_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._electron_energy_cut.string.assign(value)

    @property
    def muon_energy_cut(self):
        """Low energy cut for muons (GeV)"""
        return str(self._muon_energy_cut)

    @muon_energy_cut.setter
    def muon_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for muon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._muon_energy_cut.string.assign(value)

    @property
    def meson_energy_cut(self):
        """Low energy cut for mesons (GeV)"""
        return str(self._meson_energy_cut)

    @meson_energy_cut.setter
    def meson_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for meson_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._meson_energy_cut.string.assign(value)

    @property
    def nucleon_energy_cut(self):
        """Low energy cut for nucleons (GeV)"""
        return str(self._nucleon_energy_cut)

    @nucleon_energy_cut.setter
    def nucleon_energy_cut(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for nucleon_energy_cut {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._nucleon_energy_cut.string.assign(value)

    @property
    def other_parameters(self):
        """Other parameters"""
        return str(self._other_parameters)

    @other_parameters.setter
    def other_parameters(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(
                f"Incorrect type for other_parameters {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._other_parameters.string.assign(value)
