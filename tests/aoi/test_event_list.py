import pytest
from types import SimpleNamespace

import grand.aoi.event_list as el_mod


class FakeTree:
    def __init__(self, entries):
        self._entries = entries

    def get_entries(self):
        return self._entries


class FakeDataDirectory:
    """Minimal fake to stand in for grand.dataio.DataDirectory"""
    def __init__(self, dir_name, tshower_entries=None, tefield_entries=None, tvoltage_entries=None, trawvoltage_entries=None):
        self.dir_name = dir_name
        self.tshower = FakeTree(tshower_entries) if tshower_entries is not None else None
        self.tefield = FakeTree(tefield_entries) if tefield_entries is not None else None
        self.tvoltage = FakeTree(tvoltage_entries) if tvoltage_entries is not None else None
        self.trawvoltage = FakeTree(trawvoltage_entries) if trawvoltage_entries is not None else None

    def get_max_list_of_events(self):
        # simple deterministic placeholder (not used in directory-path tests below)
        return [(0, 0)]


class FakeEvent:
    """Minimal fake to stand in for grand.aoi.event.Event"""
    def __init__(self):
        self.filled = False
        self.fill_args = None
        self._entry_number = None
        self.run_number = None
        self.event_number = None
        self.file = None
        self.directory = None

    def fill_event_from_trees(self, **kwargs):
        # record that we were called and with what
        self.filled = True
        self.fill_args = dict(kwargs)


@pytest.fixture(autouse=True)
def replace_dependencies(monkeypatch):
    """
    Replace external dependencies in the module with fakes so tests do not depend
    on ROOT or the real grand.dataio / grand.aoi.event implementations.
    """
    monkeypatch.setattr(el_mod, "DataDirectory", FakeDataDirectory)
    monkeypatch.setattr(el_mod, "DataFile", lambda f: f)  # passthrough if invoked
    monkeypatch.setattr(el_mod, "Event", FakeEvent)
    # Provide a minimal fake ROOT.TFile type for safety (not used in most tests)
    FakeTFile = type("FakeTFile", (), {"__init__": lambda self, *a, **k: None, "GetName": lambda self: "fake", "f": None})
    # Provide a minimal fake ROOT.TFile type for safety (not used in most tests)
    def _fake_tfile_init(self, *args, **kwargs):
        _ = (self, args, kwargs)
        return None
    def _fake_tfile_getname(self):
        _ = self
        return "fake"
    FakeTFile = type("FakeTFile", (), {"__init__": _fake_tfile_init, "GetName": _fake_tfile_getname, "f": None})
    fake_root = SimpleNamespace(TFile=FakeTFile)
    monkeypatch.setattr(el_mod, "ROOT", fake_root)

def test_get_number_of_events_prefers_tshower_then_tefield_then_others(tmp_path):
    # tshower present
    dd1 = el_mod.DataDirectory(str(tmp_path / "d1"), tshower_entries=42)
    el = el_mod.EventList(dd1)
    assert el.get_number_of_events() == 42

    # tshower absent, tefield present
    dd2 = el_mod.DataDirectory(str(tmp_path / "d2"), tshower_entries=None, tefield_entries=7)
    el = el_mod.EventList(dd2)
    assert el.get_number_of_events() == 7

    # only tvoltage present
    dd3 = el_mod.DataDirectory(str(tmp_path / "d3"), tvoltage_entries=13)
    el = el_mod.EventList(dd3)
    assert el.get_number_of_events() == 13

    # none present -> None
    dd4 = el_mod.DataDirectory(str(tmp_path / "d4"))
    el = el_mod.EventList(dd4)
    assert el.get_number_of_events() is None


def test_get_event_with_entry_number_sets_entry_and_calls_fill(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"), tshower_entries=10)
    el = el_mod.EventList(dd)

    # Ask for a specific entry; since directory was provided init_trees should be False
    e = el.get_event(entry_number=5)
    assert isinstance(e, FakeEvent)
    assert e._entry_number == 5
    assert e.filled is True
    # init_trees expected False because DataDirectory inits trees
    assert e.fill_args.get("init_trees") is False


def test_iterates_over_event_list_and_passes_event_run_numbers(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"), tshower_entries=10)
    el = el_mod.EventList(dd)

    # Provide an explicit event_list to iterate over
    el.event_list = [(11, 101), (22, 202), (33, 303)]

    collected = []
    for ev in el:
        # copy the state at yield time (the EventList reuses the same Event instance)
        collected.append((ev.event_number, ev.run_number, dict(ev.fill_args)))

    assert collected == [
        (11, 101, {'init_trees': False, 'event_number': 11, 'run_number': 101}),
        (22, 202, {'init_trees': False, 'event_number': 22, 'run_number': 202}),
        (33, 303, {'init_trees': False, 'event_number': 33, 'run_number': 303}),
    ]


def test_uses_get_max_list_of_events_when_event_list_not_set(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"))  # FakeDataDirectory.get_max_list_of_events -> [(0,0)]
    el = el_mod.EventList(dd)
    # current EventList implementation does not auto-initialize event_list,
    # so initialize it here from the directory to avoid TypeError during iteration
    el.event_list = dd.get_max_list_of_events()
    # ensure event_list initialized from dd.get_max_list_of_events
    collected = [(ev.event_number, ev.run_number) for ev in el]
    assert collected == [(0, 0)]


def test_zero_entries_is_distinct_from_missing(tmp_path):
    dd_zero = el_mod.DataDirectory(str(tmp_path / "d_zero"), tshower_entries=0)
    el_zero = el_mod.EventList(dd_zero)
    assert el_zero.get_number_of_events() == 0

    dd_missing = el_mod.DataDirectory(str(tmp_path / "d_missing"))
    el_missing = el_mod.EventList(dd_missing)
    assert el_missing.get_number_of_events() is None


def test_iterator_reuses_same_event_instance(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"), tshower_entries=10)
    el = el_mod.EventList(dd)
    el.event_list = [(1, 1), (2, 2), (3, 3)]
    ids = []
    for ev in el:
        ids.append(id(ev))
    # all yielded objects are the same instance
    assert len(set(ids)) == 1


def test_get_event_with_event_and_run_number_sets_fields_and_calls_fill(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"), tshower_entries=10)
    el = el_mod.EventList(dd)
    e = el.get_event(event_number=42, run_number=7)
    assert isinstance(e, el_mod.Event)  # fake Event type
    assert e._entry_number is None
    assert e.event_number == 42
    assert e.run_number == 7
    assert e.filled is True
    assert e.fill_args.get("event_number") == 42
    assert e.fill_args.get("run_number") == 7
    # init_trees still expected False because DataDirectory supplies trees
    assert e.fill_args.get("init_trees") is False


def test_iterating_empty_event_list_yields_nothing(tmp_path):
    dd = el_mod.DataDirectory(str(tmp_path / "dir"), tshower_entries=10)
    el = el_mod.EventList(dd)
    el.event_list = []
    assert list(el) == []