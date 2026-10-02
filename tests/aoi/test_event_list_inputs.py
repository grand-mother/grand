# -*- coding: utf-8 -*-
r"""``EventList`` reads the same events whatever form its input takes.

It accepts a directory name, a file name, an open ``ROOT.TFile`` or a
``DataDirectory``.  Only the directory name used to work fully:

- a file name: ``get_number_of_events()`` wrapped the ``DataFile`` in another
  ``DataFile`` and raised ``TypeError``; ``get_event()`` raised
  ``AttributeError`` on ``trun.du_id`` when the file held no run tree;
- a ``ROOT.TFile``: it was never wrapped, so ``get_event()`` raised
  ``AttributeError`` on ``.f``, and ``event_list`` stayed ``None``;
- a ``DataDirectory``: ``event_list`` stayed ``None``, so iterating raised
  ``TypeError`` (the gap left by issue #94).
"""

import contextlib
import io
import pathlib

import pytest

SAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
EFIELD = SAMPLE / "efield_1618-13790_L0_0000.root"

pytestmark = pytest.mark.skipif(not EFIELD.is_file(), reason="the sample run is not present")

#: The events the sample holds, and the number of antennas of each.
EVENTS = {(13790, 1): 44, (1618, 1): 5}


def _inputs():
    import ROOT

    from grand.dataio import DataDirectory
    return {
        "directory name": lambda: str(SAMPLE),
        "file name": lambda: str(EFIELD),
        "ROOT.TFile": lambda: ROOT.TFile(str(EFIELD)),
        "DataDirectory": lambda: DataDirectory(str(SAMPLE)),
    }


@pytest.mark.parametrize("kind", ["directory name", "file name", "ROOT.TFile", "DataDirectory"])
def test_every_input_form_lists_counts_and_iterates_the_events(kind):
    r"""The same two events, with their efield traces, from each form."""
    from grand.aoi.event_list import EventList

    with contextlib.redirect_stdout(io.StringIO()):
        events = EventList(_inputs()[kind]())
        assert sorted((int(e), int(r)) for e, r in events.event_list) == sorted(EVENTS)
        assert events.get_number_of_events() == len(EVENTS)
        seen = {(int(e.event_number), int(e.run_number)): len(e.efields) for e in events}
    assert seen == EVENTS


def test_a_file_without_run_tree_gives_events_without_antennas(caplog):
    r"""An efield file alone has no antenna positions: none, not a crash."""
    from grand.aoi.event_list import EventList

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), caplog.at_level("WARNING", logger="grand.aoi.event"):
        event = EventList(str(EFIELD)).get_event(event_number=1618, run_number=1)
    assert len(event.efields) == 5
    assert event.antennas == []
    # Logged since #256, not printed
    assert "Antenna positions will not be available" in caplog.text


NOISE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common" / "LongNoiseTraces"
         / "noice_traces_merged_gp13_2024_02_night_datafiles_410-415.root")


@pytest.mark.skipif(not NOISE.is_file(), reason="the measured noise sample is not present")
@pytest.mark.parametrize("raw", [False, True])
def test_a_file_without_traces_it_reads_says_so(raw):
    """A file holding only ADC counts (TADC) failed with a bare IndexError."""
    from grand.aoi import EventList

    events = EventList(str(NOISE), use_trawvoltage=raw)
    with pytest.raises(ValueError, match=r"GRANDlib: .*TADC.*DataFile"):
        next(iter(events))
