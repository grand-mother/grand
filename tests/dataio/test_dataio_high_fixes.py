# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to High issues in ``grand.dataio``.

Each test names its GitHub issue and fails without the fix.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
AOI = ROOT / "examples" / "analysis" / "reconstructed_events_AOI"


def test_data_directory_keeps_every_file_of_a_level():
    r"""#195: files of one level with names of different lengths replaced each other."""
    from grand.dataio import DataDirectory

    d = DataDirectory(str(AOI))
    assert len(list(AOI.glob("shower_*_L1_*.root"))) == 10
    assert d.tshower.get_number_of_entries() == 10
    assert len(d.tshower.get_list_of_events()) == 10


def _three_showers(path):
    from grand.dataio import TShower

    t = TShower(str(path))
    for ev, zenith in ((1, 10.0), (2, 20.0), (3, 30.0)):
        t.run_number = 1
        t.event_number = ev
        t.zenith = zenith
        t.fill()
    t.write()
    return t


def test_listing_and_drawing_keep_the_loaded_entry(tmp_path):
    r"""#196: get_list_of_events() and draw() left another entry's values in some fields."""
    t = _three_showers(tmp_path / "s.root")
    t.get_entry(0)
    assert t.get_list_of_events() == [(1, 1), (2, 1), (3, 1)]
    assert (t.event_number, t.zenith) == (1, 10.0)
    t.draw("zenith", "", "goff")
    assert (t.event_number, t.zenith) == (1, 10.0)
    t.stop_using()


def test_listing_keeps_values_set_for_the_next_fill(tmp_path):
    r"""#196: values set for the next fill() were replaced by the last entry's."""
    t = _three_showers(tmp_path / "s.root")
    t.run_number = 1
    t.event_number = 4
    t.zenith = 40.0
    t.get_list_of_events()
    assert (t.event_number, t.zenith) == (4, 40.0)
    t.fill()
    t.write()
    assert t.get_list_of_events() == [(1, 1), (2, 1), (3, 1), (4, 1)]
    t.stop_using()
