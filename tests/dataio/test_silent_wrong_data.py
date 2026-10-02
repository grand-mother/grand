# -*- coding: utf-8 -*-
r"""#236: dataio returned stale or zeroed data without an error.

Reopening a path gave the file that was open before it was replaced or
removed; a tree without its run/event numbers read as zeros; a folder of
unrecognised names failed later on ``None``; an absent analysis level, a typo
such as ``trunk`` and ``get_event('x')`` were not reported as such.
"""

import os
import pathlib
import shutil

import numpy as np
import pytest
import ROOT

from grand.dataio import DataDirectory, TRun, TShower

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT_DIR / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def _shower(path, event):
    tree = TShower(str(path))
    tree.run_number, tree.event_number = 1, event
    tree.fill()
    tree.write()
    tree.stop_using()


def test_a_replaced_file_is_read_afresh(tmp_path):
    path = tmp_path / "s.root"
    _shower(path, 479)
    held = TShower(str(path))
    held.get_entry(0)
    replacement = tmp_path / "other.root"
    _shower(replacement, 17)
    os.replace(replacement, path)

    again = TShower(str(path))
    again.get_entry(0)
    assert int(again.event_number) == 17
    again.stop_using()
    held.stop_using()


def test_a_tree_without_its_numbers_is_refused(tmp_path):
    path = tmp_path / "bare.root"
    f = ROOT.TFile(str(path), "recreate")
    tree = ROOT.TTree("tshower", "tshower")
    value = np.zeros(1, dtype=np.float32)
    tree.Branch("energy_primary", value, "energy_primary/F")
    tree.Fill()
    tree.Write()
    f.Close()
    with pytest.raises(ValueError, match="no run_number or event_number branch"):
        TShower(str(path))


def test_a_folder_with_no_recognised_file_is_refused(tmp_path):
    from grand.aoi.event_list import EventList

    shutil.copy(SAMPLE / "shower_1618-13790_L0_0000.root", tmp_path / "foo.root")
    with pytest.warns(Warning, match="foo.root"):
        directory = DataDirectory(str(tmp_path))
    assert directory.unrecognised_files == [str(tmp_path / "foo.root")]
    with pytest.warns(Warning), pytest.raises(FileNotFoundError, match="no GRAND event files recognized.*foo.root"):
        EventList(str(tmp_path))


@pytest.mark.parametrize("level", [7, -5])
def test_an_absent_or_invalid_level_is_refused(level):
    with pytest.raises(ValueError, match="analysis level|analysis_level"):
        DataDirectory(str(SAMPLE), analysis_level=level)


def test_a_misspelt_tree_name_raises():
    directory = DataDirectory(str(SAMPLE))
    assert directory.tvoltage_l5 is None            # absent trees still read as None
    with pytest.raises(AttributeError, match="trunk"):
        directory.trunk


def test_non_integer_event_and_run_numbers_are_refused():
    with TShower(str(SAMPLE / "shower_1618-13790_L0_0000.root")) as shower:
        with pytest.raises(TypeError, match="ev_no must be an integer"):
            shower.get_event("x")
    with TRun(str(SAMPLE / "run_1_L0_0000.root")) as run:
        with pytest.raises(TypeError, match="run_no must be an integer"):
            run.get_run("x")


def test_an_unreadable_file_says_so(tmp_path, monkeypatch):
    path = tmp_path / "s.root"
    shutil.copy(SAMPLE / "shower_1618-13790_L0_0000.root", path)
    real = os.access
    monkeypatch.setattr(os, "access", lambda p, mode: False if os.fspath(p) == str(path) else real(p, mode))
    with pytest.raises(PermissionError, match="permission denied"):
        TShower(str(path))
