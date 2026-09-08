import inspect
import pathlib

import numpy as np
import pytest
from grand.aoi.antenna import Antenna
from grand.aoi.event import Event
from grand.aoi.timetrace import Voltage

def test_event_defaults():
    """Test default initialization types and shapes for Event."""
    e = Event()

    # Basic scalar fields
    assert e.event_number is None
    assert e.run_number is None
    assert isinstance(e.is_reconstructed, bool)
    assert isinstance(e.is_wave, bool)
    assert isinstance(e.is_eas, bool)

    # Time vector default
    assert isinstance(e.t_vector, np.ndarray)
    assert e.t_vector.shape == (1,)
    assert e.t_vector.dtype == np.float32

    # Origin geoid defaults
    assert e.origin_geoid is not None
    for attr in ['x', 'y', 'z']:
        arr = getattr(e.origin_geoid, attr)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1,)
        assert arr.dtype == np.float64

    # Internal placeholders
    assert e.antennas is None
    assert e.voltages is None
    assert e.efields is None

    # Other defaults
    assert isinstance(e.run_mode, (np.uint32, int))
    # file is stringified
    assert isinstance(e.file, str)


def test_event_origin_geoid_setter_getter():
    """Test that setting origin_geoid works correctly."""
    e = Event()
    new_origin = (np.array([10.0], dtype=np.float64),
                  np.array([20.0], dtype=np.float64),
                  np.array([30.0], dtype=np.float64))
    e.origin_geoid = new_origin

    assert isinstance(e.origin_geoid, type(e._origin_geoid))
    for val in [e.origin_geoid.x, e.origin_geoid.y, e.origin_geoid.z]:
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert val.dtype == np.float64

    assert np.allclose(e.origin_geoid.x, new_origin[0])
    assert np.allclose(e.origin_geoid.y, new_origin[1])
    assert np.allclose(e.origin_geoid.z, new_origin[2])


def test_fill_t_vector_rectangular():
    """fill_t_vector with same-length t_vectors (rectangular array)."""
    e = Event()

    v1 = Voltage()
    v1.t_vector = np.array([0, 2, 4], dtype=np.int64)

    v2 = Voltage()
    v2.t_vector = np.array([0, 2, 6], dtype=np.int64)

    e.voltages = [v1, v2]

    e.fill_t_vector(resolution=1)
    assert isinstance(e.t_vector, np.ndarray)
    assert np.array_equal(e.t_vector, np.arange(0, 6 + 1, 1))


def test_fill_t_vector_irregular():
    """fill_t_vector with different-length t_vectors (object array path)."""
    e = Event()

    v1 = Voltage()
    v1.t_vector = np.array([0, 3], dtype=np.int64)

    v2 = Voltage()
    v2.t_vector = np.array([1, 2, 5], dtype=np.int64)

    e.voltages = [v1, v2]

    e.fill_t_vector(resolution=1)
    assert np.array_equal(e.t_vector, np.arange(0, 5 + 1, 1))


def test_get_voltage_at_time_stacks_values():
    """get_voltage_at_time returns stacked values from all voltages."""
    e = Event()

    v1 = Voltage()
    v2 = Voltage()

    # Monkeypatch instance methods to avoid building full trace structures
    v1.get_value_at_time = lambda t: np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2.get_value_at_time = lambda t: np.array([4.0, 5.0, 6.0], dtype=np.float32)

    e.voltages = [v1, v2]
    out = e.get_voltage_at_time(123456789)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1.0, 2.0, 3.0])
    assert np.allclose(out[1], [4.0, 5.0, 6.0])


def test_get_hilbert_voltage_at_time_stacks_values():
    """get_hilbert_voltage_at_time stacks hilbert values from all voltages."""
    e = Event()

    v1 = Voltage()
    v2 = Voltage()

    v1.get_hilbert_value_at_time = lambda t: np.array([10.0, 20.0, 30.0], dtype=np.float32)
    v2.get_hilbert_value_at_time = lambda t: np.array([40.0, 50.0, 60.0], dtype=np.float32)

    e.voltages = [v1, v2]
    out = e.get_hilbert_voltage_at_time(42)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [10.0, 20.0, 30.0])
    assert np.allclose(out[1], [40.0, 50.0, 60.0])


class DummyTRun:
    def __init__(self):
        self.last_call = None
        self.run_mode = np.uint32(7)
        self.data_source = "detector"
        self.data_generator = "GEN"
        self.data_generator_version = "v1.2.3"
        self.site = "DummySite"
        self.origin_geoid = (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        )
        self.t_bin_size = np.array([4], dtype=np.int32)
        self.site_layout = "star_shape"

    def get_run(self, run_number):
        self.last_call = ("get_run", run_number)
        return 1

    def get_entry(self, entry_number):
        self.last_call = ("get_entry", entry_number)
        return 1


class DummyTRunRawVoltage:
    def __init__(self):
        self.last_call = None

    def get_run(self, run_number):
        self.last_call = ("get_run", run_number)
        return 1

    def get_entry(self, entry_number):
        self.last_call = ("get_entry", entry_number)
        return 1


def test_fill_event_from_runtree_copies_fields_and_starshape():
    """fill_event_from_runtree reads from provided trun and copies fields."""
    e = Event()
    e.trun = DummyTRun()

    # Case: default, run_entry_number is None, run_number is None -> uses entry 0
    ret = e.fill_event_from_runtree(run_entry_number=None)
    assert ret == 1
    assert e.trun.last_call == ("get_entry", 0)

    # Fields copied correctly
    assert e.run_mode == np.uint32(7)
    assert e.data_source == "detector"
    assert e.data_generator == "GEN"
    assert e.data_generator_version == "v1.2.3"
    assert e.site == "DummySite"
    assert e._t_bin_size == 4
    assert e.is_starshape is True

    # origin_geoid copied
    for attr, expected in zip(['x', 'y', 'z'], e.trun.origin_geoid):
        arr = getattr(e.origin_geoid, attr)
        assert np.allclose(arr, expected)


def test_fill_event_from_runrawvoltagetree_starshape_uses_entry_number():
    """fill_event_from_runrawvoltagetree uses _entry_number if starshape and run unset."""
    e = Event()
    e.is_starshape = True
    e._entry_number = 5
    e.trunrawvoltage = DummyTRunRawVoltage()

    ret = e.fill_event_from_runrawvoltagetree(run_entry_number=None)
    assert ret == 1
    assert e.trunrawvoltage.last_call == ("get_entry", 5)


def test_fill_run_tree_raises_treeexists_when_trun_present():
    """fill_run_tree should raise TreeExists if trun already initialised and not overwriting."""
    e = Event()
    e.trun = object()  # any non-None sentinel
    with pytest.raises(Exception):
        e.fill_run_tree(overwrite=False)


def test_fill_voltage_tree_raises_treeexists_when_tvoltage_present():
    """fill_voltage_tree should raise TreeExists if tvoltage already initialised and not overwriting."""
    e = Event()
    e.tvoltage = object()
    with pytest.raises(Exception):
        e.fill_voltage_tree(overwrite=False)


def test_fill_efield_tree_raises_treeexists_when_tefield_present():
    """fill_efield_tree should raise TreeExists if tefield already initialised and not overwriting."""
    e = Event()
    e.tefield = object()
    with pytest.raises(Exception):
        e.fill_efield_tree(overwrite=False)

# --------------------------------------------------------------------------
# The GP300/GP80/GP13 workaround. No site in any file we hold matches those
# names, so nothing below can be reached through real data -- these fakes are
# the only thing standing between the workaround and a silent regression.
# --------------------------------------------------------------------------


class _GPSWorkaroundEntered(Exception):
    """Signals that fill_antennas took the GPS branch."""


class DummyAntennaTree:
    """Stands in for tefield inside fill_antennas.

    The GPS branch opens with a call to `draw`, so raising there pinpoints the
    moment the branch is entered without having to fake the coordinate
    transforms that follow it.
    """

    def __init__(self, n_event_dus=2, du_id=None):
        self._n = n_event_dus
        self.du_id = du_id if du_id is not None else []

    def draw(self, *args, **kwargs):
        raise _GPSWorkaroundEntered()

    def get_dus_indices_in_run(self, trun):
        return list(range(self._n))

    def get_event(self, event_number, run_number):
        return 1

    def get_entry(self, entry_number):
        return 1


class DummyRunForAntennas:
    """Stands in for trun inside fill_antennas."""

    def __init__(self, n_dus=3):
        self.du_id = list(range(100, 100 + n_dus))
        self.du_xyz = [[float(i), float(i) + 0.5, float(i) + 0.25]
                       for i in range(n_dus)]
        self.du_tilt = [[0.0, 0.0] for _ in range(n_dus)]


def _event_at_site(site, n_event_dus=2, du_id=None):
    """Builds an Event whose only purpose is to reach fill_antennas."""
    e = Event()
    e.site = site
    e.trun = DummyRunForAntennas()
    e.tefield = DummyAntennaTree(n_event_dus, du_id)
    e.tvoltage = None
    return e


@pytest.mark.parametrize("site", ["GP300", "GP80", "GP13"])
def test_workaround_runs_at_every_gp_site_when_asked(site):
    """All three site names opt into the workaround."""
    e = _event_at_site(site)
    with pytest.raises(_GPSWorkaroundEntered):
        e.fill_antennas(gp300_workaround=True)


@pytest.mark.parametrize("site", ["GP300", "GP80", "GP13"])
def test_the_flag_switches_the_workaround_off_at_every_gp_site(site):
    """`and` binds tighter than `or`.

    Written without parentheses, the condition gated only the GP300 arm, and
    GP80 and GP13 took the workaround whatever the caller asked for.
    """
    e = _event_at_site(site)
    e.fill_antennas(gp300_workaround=False)
    assert len(e.antennas) == 2
    assert e._all_antennas_key == ("run", site)


def test_a_site_outside_the_three_never_takes_the_workaround():
    e = _event_at_site("Xiaodushan")
    e.fill_antennas(gp300_workaround=True)
    assert e._all_antennas_key == ("run", "Xiaodushan")


def test_run_built_positions_do_not_suppress_the_gps_calculation():
    """The two branches fill `_all_antennas` from different sources.

    The GPS branch treats a non-empty dict as work already done, so a dict left
    behind by the ordinary branch used to make it skip and hand back positions
    taken from `du_xyz` instead of from GPS.
    """
    e = _event_at_site("GP13")
    e.fill_antennas(gp300_workaround=False)
    assert e._all_antennas, "the ordinary branch should have filled the dict"
    assert e._all_antennas_key == ("run", "GP13")

    with pytest.raises(_GPSWorkaroundEntered):
        e.fill_antennas(gp300_workaround=True)


def test_gps_built_positions_are_reused_for_the_same_site():
    """The cache still does its job: the GPS draw is expensive and runs once."""
    e = _event_at_site("GP13", du_id=[100])
    cached = Antenna()
    cached.id = 100
    cached.position.x, cached.position.y, cached.position.z = 1.0, 2.0, 3.0
    cached.tilt.x, cached.tilt.y = 0.0, 0.0
    e._all_antennas = {100: cached}
    e._all_antennas_key = ("gps", "GP13")

    e.fill_antennas(gp300_workaround=True)

    assert len(e.antennas) == 1
    assert e.antennas[0].position.x == 1.0


def test_fill_event_from_trees_forwards_the_workaround_flag():
    """The parameter used to be documented, accepted, and then ignored."""
    source = inspect.getsource(Event.fill_event_from_trees)
    assert "self.fill_antennas(gp300_workaround=gp300_workaround)" in source
    assert "self.fill_antennas(gp300_workaround=True)" not in source


# --------------------------------------------------------------------------
# Choosing the analysis level of the electric-field tree. The level is not in
# the tree name -- every file calls its tree "tefield" and records the level
# in the tree's metadata -- so asking for one is really a statement about
# which file is open, and the code has to check rather than assume.
# --------------------------------------------------------------------------

SIM_DIR = pathlib.Path(__file__).resolve().parents[2] / (
    "sim2root/Common/sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS_0000")
EFIELD_L1 = SIM_DIR / "efield_5388-23832_L1_0000.root"
RUN_L1 = SIM_DIR / "run_0_L1_0000.root"


def _event_on_the_level_one_file():
    """An Event wired to the level-1 efield file, without a directory."""
    import ROOT

    e = Event()
    e.file_trun = ROOT.TFile.Open(str(RUN_L1))
    e.file_tefield = ROOT.TFile.Open(str(EFIELD_L1))
    return e


@pytest.mark.skipif(not EFIELD_L1.exists() or not RUN_L1.exists(),
                    reason="the L1 simulation files are not present")
def test_requesting_the_level_the_open_file_holds_is_accepted():
    e = _event_on_the_level_one_file()
    e.fill_event_from_trees(entry_number=0, tefield_level=1)
    assert e.tefield is not None
    assert e.tefield.analysis_level == 1
    assert e.tefield_level == 1


@pytest.mark.skipif(not EFIELD_L1.exists() or not RUN_L1.exists(),
                    reason="the L1 simulation files are not present")
def test_requesting_a_level_the_open_file_does_not_hold_is_refused():
    """Reading a different level than the one asked for is invisible to the
    caller, and leaving the tree unset is not a state the rest of Event
    supports -- it indexes the filled traces and assumes one survived. So the
    request is refused outright."""
    e = _event_on_the_level_one_file()
    with pytest.raises(ValueError, match="analysis level 1, not the requested 0"):
        e.fill_event_from_trees(entry_number=0, tefield_level=0)


@pytest.mark.skipif(not EFIELD_L1.exists() or not RUN_L1.exists(),
                    reason="the L1 simulation files are not present")
def test_requesting_no_level_reads_whatever_is_there():
    e = _event_on_the_level_one_file()
    e.fill_event_from_trees(entry_number=0)
    assert e.tefield is not None
    assert e.tefield.analysis_level == 1


def test_an_event_with_no_efield_and_no_voltage_has_no_antennas():
    """`event_dus_indices` was left unbound when neither tree was present, so
    the loop over it raised instead of yielding an empty antenna list. The
    level check above made that reachable by refusing a mismatched tree."""
    e = Event()
    e.site = "Xiaodushan"
    e.trun = DummyRunForAntennas()
    e.tefield = None
    e.tvoltage = None

    e.fill_antennas(gp300_workaround=True)

    assert e.antennas == []
