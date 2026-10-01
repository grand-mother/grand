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


def test_write_does_not_silently_replace_a_tree(tmp_path):
    r"""#197: a new tree written into a file holding one of that name replaced it, even
    with overwrite=False; overwrite=True also wiped every other tree in the file."""
    import pytest
    import ROOT

    from grand.dataio import TRun, TShower

    path = str(tmp_path / "f.root")
    r = TRun()
    r.run_number = 1
    r.fill()
    r.write(path)
    r.stop_using()
    for ev in (10, 50):
        t = TShower()
        t.run_number = 1
        t.event_number = ev
        t.fill()
        if ev == 10:
            t.write(path)
        else:
            with pytest.raises(FileExistsError, match="already holds a tshower tree"):
                t.write(path)
            assert TShower(path).get_list_of_events() == [(10, 1)]
            t.write(path, overwrite=True)
        t.stop_using()
    assert TShower(path).get_list_of_events() == [(50, 1)]
    f = ROOT.TFile(path)
    assert sorted(k.GetName() for k in f.GetListOfKeys()) == ["trun", "tshower"]
    f.Close()


def test_writing_to_another_file_copies_the_tree(tmp_path):
    r"""#198: write("other.root") on a tree stored in a file wrote a corrupt copy."""
    from grand.dataio import TShower

    first, second = str(tmp_path / "w1.root"), str(tmp_path / "w2.root")
    t = TShower(first)
    for ev in (1, 2, 3):
        t.run_number = 1
        t.event_number = ev
        t.zenith = 40.0 + ev
        t.fill()
        if ev == 2:
            t.write()                 # event 3 is filled but not yet written
    t.write(second)
    assert t.event_number == 3
    t.write()
    t.stop_using()
    for name in (first, second):
        s = TShower(name)
        assert s.get_list_of_events() == [(1, 1), (2, 1), (3, 1)]
        s.get_entry(2)
        assert s.zenith == 43.0
        s.stop_using()


def test_du_indices_follow_the_event_order():
    r"""#199: the indices came in the run's order, pairing positions with the wrong traces."""
    import pytest

    from grand.dataio import TEfield, TRun

    r = TRun()
    r.du_id = [60003, 60000, 60002, 60001, 60009]
    e = TEfield()
    e.du_id = [60000, 60001, 60002]
    assert e.get_dus_indices_in_run(r).tolist() == [1, 3, 2]
    e.du_id = [60001, 60004]
    with pytest.raises(ValueError, match=r"units \[60004\]"):
        e.get_dus_indices_in_run(r)
    r.stop_using()
    e.stop_using()


def test_filled_entries_are_written_or_reported(tmp_path):
    r"""#275: entries filled but not written were dropped silently at the end of a with
    block or at stop_using()."""
    import warnings

    import pytest

    from grand.basis.validate import GRANDlibWarning
    from grand.dataio import TADC

    path = str(tmp_path / "w.root")
    with TADC(path) as t:
        t.run_number = 1
        t.event_number = 1
        t.fill()
    t = TADC(path)
    assert t.get_number_of_entries() == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t.stop_using()                       # nothing pending: no warning
    t = TADC(str(tmp_path / "x.root"))
    t.run_number = 1
    t.event_number = 1
    t.fill()
    with pytest.warns(GRANDlibWarning, match="1 entries were filled but not written"):
        t.stop_using()
    with pytest.warns(GRANDlibWarning, match="filled but not written"):
        with pytest.raises(RuntimeError):
            with TADC(str(tmp_path / "y.root")) as t:
                t.run_number = 1
                t.event_number = 1
                t.fill()
                raise RuntimeError("stop")


SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

USE_AFTER_CLOSE = [
    "e = TEfield(F); e.close_file(); e.get_entry(0)",
    "a = TEfield(F); b = TEfield(F); a.close_file(); b.get_entry(1)",
    "df = DataFile(F); df.close(); df.tefield.get_entry(0)",
    "d = DataDirectory(D); d.close(); d.tefield.get_entry(0)",
    "e = TEfield(N); e.run_number = 1; e.event_number = 1; e.fill(); e.write(); "
    "e.close_file(); e.event_number = 2; e.fill()",
]


def test_using_a_tree_after_its_file_closed_raises(tmp_path):
    r"""#274: each of these crashed the interpreter (exit 129, no traceback)."""
    import shutil
    import subprocess
    import sys

    data = tmp_path / "d"
    data.mkdir()
    for f in SAMPLE.glob("*.root"):
        shutil.copy(f, data)
    efield = next(data.glob("efield_*_L0_*.root"))
    for snippet in USE_AFTER_CLOSE:
        code = ("from grand.dataio import TEfield, DataFile, DataDirectory\n"
                "F, D, N = %r, %r, %r\n%s\n" % (str(efield), str(data), str(tmp_path / "n.root"),
                                               snippet))
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 1, (snippet, done.returncode, done.stderr[-800:])
        assert "this tree's file was closed" in done.stderr, snippet
