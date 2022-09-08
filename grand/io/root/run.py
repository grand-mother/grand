
from grand.io.root.base import *


## A mother class for classes with Run values
@dataclass
class MotherRunTree(DataTree):
    """A mother class for classes with Run values"""
    _run_number: np.ndarray = np.zeros(1, np.uint32)

    @property
    def run_number(self):
        """The run number for this tree entry"""
        return self._run_number[0]

    @run_number.setter
    def run_number(self, val: np.uint32) -> None:
        self._run_number[0] = val

    def fill(self):
        """Adds the current variable values as a new event to the tree"""
        # If the current run_number and event_number already exist, raise an exception
        if not self.is_unique_event():
            raise NotUniqueEvent(f"A run with run_number={self.run_number} already exists in the TTree.")

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
        # Try to get the run from the tree
        res = self._tree.GetEntryWithIndex(run_no)
        # If no such entry, return
        if res==0 or res==-1:
            print(f"No run with run number {run_no}. Please provide a proper number.")
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

## A class wrapping around a TTree holding values common for the whole run
@dataclass
class RunTree(MotherRunTree):
    """A class wrapping around a TTree holding values common for the whole run"""
    _tree_name: str = "trun"

    ## Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment
    _run_mode: np.ndarray = np.zeros(1, np.uint32)
    ## Run's first event
    _first_event: np.ndarray = np.zeros(1, np.uint32)
    ## First event time
    _first_event_time: np.ndarray = np.zeros(1, np.uint32)
    ## Run's last event
    _last_event: np.ndarray = np.zeros(1, np.uint32)
    ## Last event time
    _last_event_time: np.ndarray = np.zeros(1, np.uint32)

    # These are not from the hardware
    ## Data source: detector, simulation, other
    _data_source: StdString = StdString("detector")
    ## Data generator: gtot (in this case)
    _data_generator: StdString = StdString("GRANDlib")
    ## Generator version: gtot version (in this case)
    _data_generator_version: StdString = StdString("0.1.0")
    ## Site name
    # _site: StdVectorList("string") = StdVectorList("string")
    _site: StdString = StdString("")
    ## Site longitude
    _site_long: np.ndarray = np.zeros(1, np.float32)
    ## Site latitude
    _site_lat: np.ndarray = np.zeros(1, np.float32)
    ## Origin of the coordinate system used for the array
    _origin_geoid: np.ndarray = np.zeros(3, np.float32)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def run_mode(self):
        """Run mode - calibration/test/physics. ToDo: should get enum description for that, but I don't think it exists at the moment"""
        return self._run_mode[0]

    @run_mode.setter
    def run_mode(self, value: np.uint32) -> None:
        self._run_mode[0] = value

    @property
    def first_event(self):
        """Run's first event"""
        return self._first_event[0]

    @first_event.setter
    def first_event(self, value: np.uint32) -> None:
        self._first_event[0] = value

    @property
    def first_event_time(self):
        """First event time"""
        return self._first_event_time[0]

    @first_event_time.setter
    def first_event_time(self, value: np.uint32) -> None:
        self._first_event_time[0] = value

    @property
    def last_event(self):
        """Run's last event"""
        return self._last_event[0]

    @last_event.setter
    def last_event(self, value: np.uint32) -> None:
        self._last_event[0] = value

    @property
    def last_event_time(self):
        """Last event time"""
        return self._last_event_time[0]

    @last_event_time.setter
    def last_event_time(self, value: np.uint32) -> None:
        self._last_event_time[0] = value

    @property
    def data_source(self):
        """Data source: detector, simulation, other"""
        return str(self._data_source)

    @data_source.setter
    def data_source(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_source.string.assign(value)

    @property
    def data_generator(self):
        """Data generator: gtot (in this case)"""
        return str(self._data_generator)

    @data_generator.setter
    def data_generator(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_generator.string.assign(value)

    @property
    def data_generator_version(self):
        """Generator version: gtot version (in this case)"""
        return str(self._data_generator_version)

    @data_generator_version.setter
    def data_generator_version(self, value) -> None:
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._data_generator_version.string.assign(value)

    @property
    def site(self):
        """Site name"""
        return str(self._site)

    @site.setter
    def site(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._site.string.assign(value)


    @property
    def site_long(self):
        """Site longitude"""
        return np.array(self._site_long)

    @site_long.setter
    def site_long(self, value):
        self._site_long = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("site_long", self._site_long)

    @property
    def site_lat(self):
        """Site latitude"""
        return np.array(self._site_lat)

    @site_lat.setter
    def site_lat(self, value):
        self._site_lat = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("site_lat", self._site_lat)

    @property
    def origin_geoid(self):
        """Origin of the coordinate system used for the array"""
        return np.array(self._origin_geoid)

    @origin_geoid.setter
    def origin_geoid(self, value):
        self._origin_geoid = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("origin_geoid", self._origin_geoid)


@dataclass
## The class for storing voltage simulation-only data common for a whole run
class VoltageRunSimdataTree(MotherRunTree):
    """The class for storing voltage simulation-only data common for a whole run"""
    _tree_name: str = "trunvoltagesimdata"

    _signal_sim: StdString = StdString("")  # name and model of the signal simulator

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def signal_sim(self):
        """Name and model of the signal simulator"""
        return str(self._signal_sim)

    @signal_sim.setter
    def signal_sim(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._signal_sim.string.assign(value)



@dataclass
## The class for storing Efield simulation-only data common for a whole run
class EfieldRunSimdataTree(MotherRunTree):
    """The class for storing Efield simulation-only data common for a whole run"""
    _tree_name: str = "trunefieldsimdata"

    ## Name and model of the electric field simulator
    # _field_sim: StdString = StdString("")
    ## Name of the atmospheric index of refraction model
    _refractivity_model: StdString = StdString("")
    _refractivity_model_parameters: StdVectorList("double") = StdVectorList("double")
    ## The antenna time window is defined arround a t0 that changes with the antenna, starts on t0+t_pre (thus t_pre is usually negative) and ends on t0+post
    _t_pre: np.ndarray = np.zeros(1, np.float32)
    _t_post: np.ndarray = np.zeros(1, np.float32)
    _t_bin_size: np.ndarray = np.zeros(1, np.float32)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    # @property
    # def field_sim(self):
    #     return str(self._field_sim)
    #
    # @field_sim.setter
    # def field_sim(self, value):
    #     # Not a string was given
    #     if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
    #         exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")
    #
    #     self._field_sim.string.assign(value)

    @property
    def refractivity_model(self):
        """Name of the atmospheric index of refraction model"""
        return str(self._refractivity_model)

    @refractivity_model.setter
    def refractivity_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._refractivity_model.string.assign(value)

    @property
    def refractivity_model_parameters(self):
        """Refractivity model parameters"""
        return self._refractivity_model_parameters

    @refractivity_model_parameters.setter
    def refractivity_model_parameters(self, value) -> None:
        # A list of strings was given
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # Clear the vector before setting
            self._refractivity_model_parameters.clear()
            self._refractivity_model_parameters += value
        # A vector was given
        elif isinstance(value, ROOT.vector("unsigned short")):
            self._refractivity_model_parameters._vector = value
        else:
            exit(f"Incorrect type for refractivity_model_parameters {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required.")

    @property
    def t_pre(self):
        """Starting time of antenna data collection time window. The window starts at t0+t_pre, thus t_pre is usually negative."""
        return self._t_pre[0]

    @t_pre.setter
    def t_pre(self, value):
        self._t_pre[0] = value

    @property
    def t_post(self):
        """Finishing time of antenna data collection time window. The window ends at t0+t_post."""
        return self._t_post[0]

    @t_post.setter
    def t_post(self, value):
        self._t_post[0] = value

    @property
    def t_bin_size(self):
        """Time bin size"""
        return self._t_bin_size[0]

    @t_bin_size.setter
    def t_bin_size(self, value):
        self._t_bin_size[0] = value



@dataclass
## The class for storing shower simulation-only data common for a whole run
class ShowerRunSimdataTree(MotherRunTree):
    """The class for storing shower simulation-only data common for a whole run"""
    _tree_name: str = "trunsimdata"

    _shower_sim: StdString = StdString("") # simulation program (and version) used to simulate the shower
    _rel_thin: np.ndarray = np.zeros(1, np.float32)  # relative thinning energy
    _weight_factor: np.ndarray = np.zeros(1, np.float32)  # weight factor
    _lowe_cut_e: np.ndarray = np.zeros(1, np.float32)  # low energy cut for electrons(GeV)
    _lowe_cut_gamma: np.ndarray = np.zeros(1, np.float32)  # low energy cut for gammas(GeV)
    _lowe_cut_mu: np.ndarray = np.zeros(1, np.float32)  # low energy cut for muons(GeV)
    _lowe_cut_meson: np.ndarray = np.zeros(1, np.float32)  # low energy cut for mesons(GeV)
    _lowe_cut_nucleon: np.ndarray = np.zeros(1, np.float32)  # low energy cut for nuceleons(GeV)

    def __post_init__(self):
        super().__post_init__()

        if self._tree.GetName() == "":
            self._tree.SetName(self._tree_name)
        if self._tree.GetTitle() == "":
            self._tree.SetTitle(self._tree_name)

        self.create_branches()

    @property
    def shower_sim(self):
        """Simulation program (and version) used to simulate the shower"""
        return str(self._shower_sim)

    @shower_sim.setter
    def shower_sim(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            exit(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")

        self._shower_sim.string.assign(value)

    @property
    def rel_thin(self):
        """relative thinning energy"""
        return self._rel_thin[0]

    @rel_thin.setter
    def rel_thin(self, value):
        self._rel_thin[0] = value

    @property
    def weight_factor(self):
        """weight factor"""
        return self._weight_factor[0]

    @weight_factor.setter
    def weight_factor(self, value):
        self._weight_factor[0] = value

    @property
    def lowe_cut_e(self):
        """low energy cut for electrons(GeV)"""
        return self._lowe_cut_e[0]

    @lowe_cut_e.setter
    def lowe_cut_e(self, value):
        self._lowe_cut_e[0] = value

    @property
    def lowe_cut_gamma(self):
        """low energy cut for gammas(GeV)"""
        return self._lowe_cut_gamma[0]

    @lowe_cut_gamma.setter
    def lowe_cut_gamma(self, value):
        self._lowe_cut_gamma[0] = value

    @property
    def lowe_cut_mu(self):
        """low energy cut for muons(GeV)"""
        return self._lowe_cut_mu[0]

    @lowe_cut_mu.setter
    def lowe_cut_mu(self, value):
        self._lowe_cut_mu[0] = value

    @property
    def lowe_cut_meson(self):
        """low energy cut for mesons(GeV)"""
        return self._lowe_cut_meson[0]

    @lowe_cut_meson.setter
    def lowe_cut_meson(self, value):
        self._lowe_cut_meson[0] = value

    @property
    def lowe_cut_nucleon(self):
        """low energy cut for nucleons(GeV)"""
        return self._lowe_cut_nucleon[0]

    @lowe_cut_nucleon.setter
    def lowe_cut_nucleon(self, value):
        self._lowe_cut_nucleon[0] = value

