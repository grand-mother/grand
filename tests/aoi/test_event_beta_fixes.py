# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to ``grand.aoi`` (Event, EventList).

Each test names its GitHub issue and fails without the fix.  They run on a
copy of the committed sim2root sample RUN1 (events 13790 and 1618).
"""

import pathlib
import shutil

import pytest

SAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="needs the committed RUN1 sample")


@pytest.fixture
def sample(tmp_path):
    target = tmp_path / "sample"
    shutil.copytree(SAMPLE, target)
    return target


def test_write_overwrite_keeps_other_files(sample, tmp_path):
    r"""#212: overwrite=True removed the whole output directory."""
    from grand.aoi import EventList

    out = tmp_path / "out"
    e = EventList(str(sample)).get_event(event_number=13790, run_number=1)
    e.write(out_dir=str(out))
    (out / "precious.txt").write_text("x")
    e.write(out_dir=str(out), overwrite=True)
    assert (out / "precious.txt").read_text() == "x"
    assert len(list(out.glob("efield_*.root"))) == 1


def test_write_without_a_destination_is_refused(sample):
    r"""#212: write() with no file name crashed with AttributeError."""
    from grand.aoi import EventList

    e = EventList(str(sample)).get_event(event_number=13790, run_number=1)
    with pytest.raises(ValueError, match="writing back into the files"):
        e.write()


def test_write_to_a_common_file_keeps_the_event_list_usable(sample, tmp_path):
    r"""#212: common_filename always raised TreeExists, and left the EventList reading
    showers with zenith 0 and Xmax 0."""
    from grand.aoi import EventList
    from grand.dataio import TEfield

    path = str(tmp_path / "all.root")
    events = EventList(str(sample))
    first = events.get_event(event_number=13790, run_number=1)
    first.write(common_filename=path)
    second = events.get_event(event_number=1618, run_number=1)
    assert len(second.efields) == 5
    second.write(common_filename=path)
    t = TEfield(path)
    assert t.get_list_of_events() == [(13790, 1), (1618, 1)]
    t.stop_using()
    second.write(common_filename=path, overwrite=True)
    t = TEfield(path)
    assert t.get_list_of_events() == [(1618, 1)]
    t.stop_using()
    again = events.get_event(event_number=13790, run_number=1)
    assert len(again.efields) == 44


AOI = pathlib.Path(__file__).resolve().parents[2] / "examples" / "analysis" / "reconstructed_events_AOI"


def test_iteration_honours_start_event_and_start_entry(sample):
    r"""#213: start_event and start_entry were stored but ignored."""
    from grand.aoi import EventList

    assert [int(e.event_number) for e in EventList(str(sample))] == [13790, 1618]
    assert [int(e.event_number) for e in EventList(str(sample), start_event=1618)] == [1618]
    assert [int(e.event_number) for e in EventList(str(sample), start_entry=1)] == [1618]
    with pytest.raises(ValueError, match="start_event 7 is not in the input"):
        list(EventList(str(sample), start_event=7))


def test_tefield_level_per_call(sample):
    r"""#213: a per-call tefield_level was ignored for directories, and a missing level
    gave efields=None instead of an error."""
    from grand.aoi import EventList

    events = EventList(str(sample))
    assert events.get_event(event_number=13790, run_number=1).efields[0].n_points == 2048
    assert events.get_event(event_number=13790, run_number=1,
                            tefield_level=0).efields[0].n_points == 8192
    assert events.get_event(event_number=13790, run_number=1).efields[0].n_points == 2048
    with pytest.raises(ValueError, match="no Efield tree of analysis level 2"):
        events.get_event(event_number=13790, run_number=1, tefield_level=2)


@pytest.mark.skipif(not AOI.is_dir(), reason="needs examples/analysis")
def test_raw_voltage_channels_and_tree_choice():
    r"""#213: two channels crashed with IndexError; after a use_trawvoltage call the next
    default call failed on TRawVoltage's missing 'trace'."""
    from grand.aoi import EventList

    events = EventList(str(AOI))
    with pytest.raises(ValueError, match="exactly 3 channels"):
        events.get_event(entry_number=0, use_trawvoltage=True, trawvoltage_channels=[0, 2])
    raw = events.get_event(entry_number=0, use_trawvoltage=True, trawvoltage_channels=[1, 2, 3])
    assert len(raw.voltages) == 5
    assert len(events.get_event(entry_number=0).voltages) == 5


@pytest.mark.skipif(not AOI.is_dir(), reason="needs examples/analysis")
@pytest.mark.parametrize("entry, error", [(10, IndexError), (-1, IndexError), (10**9, IndexError),
                                          (2**70, IndexError), (True, TypeError), (1.5, TypeError),
                                          ("3", TypeError)])
def test_entry_number_is_checked(entry, error):
    r"""#235: out-of-range entries failed deep in the reader; bools were taken as 1."""
    from grand.aoi import EventList

    with pytest.raises(error, match="GRANDlib: EventList.get_event"):
        EventList(str(AOI)).get_event(entry_number=entry)


@pytest.mark.skipif(not AOI.is_dir(), reason="needs examples/analysis")
def test_unusable_inputs_say_what_is_wrong(tmp_path):
    r"""#235: a shower without a run tree, a foreign tree, a closed TFile and a non-ROOT
    file all failed with errors that did not say what was wrong."""
    import array

    import ROOT

    from grand.aoi import Event, EventList
    from grand.dataio import DataFile

    shower = sorted(AOI.glob("shower_*.root"))[0]
    with pytest.raises(FileNotFoundError, match="needs the run tree"):
        EventList(str(shower)).get_event(entry_number=0)

    foreign = str(tmp_path / "foo.root")
    f = ROOT.TFile(foreign, "recreate")
    tree = ROOT.TTree("foo", "foo")
    x = array.array("i", [0])
    tree.Branch("x", x, "x/I")
    tree.Fill()
    tree.Write()
    f.Close()
    with pytest.raises(ValueError, match="holds no GRAND tree"):
        DataFile(foreign)

    closed = ROOT.TFile(str(AOI / "run_00000_L0_0000.root"))
    closed.Close()
    with pytest.raises(ValueError, match="is closed"):
        DataFile(closed)

    text = tmp_path / "text.root"
    text.write_text("hello")
    with pytest.raises(OSError, match="GRANDlib: DataFile: cannot open"):
        EventList(str(text))

    e = Event()
    e.file = AOI / "run_00000_L0_0000.root"          # a path object is accepted
    assert "run_00000_L0_0000.root" in e.file


@pytest.mark.skipif(not AOI.is_dir(), reason="needs examples/analysis")
def test_close_files_leaves_the_input_alone(tmp_path):
    r"""#234: close_files() wrote every tree, including those only read, changing the
    input files (and could hang)."""
    import hashlib

    from grand.aoi import EventList

    data = tmp_path / "aoi"
    shutil.copytree(AOI, data)
    before = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in data.glob("*.root")}
    event = EventList(str(data), use_trawvoltage=True).get_event(entry_number=0)
    event.close_files()
    after = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in data.glob("*.root")}
    assert after == before


@pytest.mark.skipif(not AOI.is_dir(), reason="needs examples/analysis")
def test_reading_an_event_exits_cleanly():
    r"""#234: a script that only read an event crashed or hung at exit in about a third of
    the runs (exit 129/139, in ROOT's cleanup)."""
    import subprocess
    import sys

    code = ("from grand.aoi import EventList\n"
            "EventList(%r, use_trawvoltage=True, trawvoltage_channels=[1, 2, 3])"
            ".get_event(entry_number=0)\n" % str(AOI))
    codes = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                            timeout=120).returncode for _ in range(8)]
    assert codes == [0] * 8
