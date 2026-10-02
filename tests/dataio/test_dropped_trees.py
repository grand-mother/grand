# -*- coding: utf-8 -*-
r"""#284: a tree dropped without stop_using() releases its file.

The registry kept every tree, and the file it opened, until stop_using(): a
loop that only dropped its trees grew by about 630 kB a file.  It now holds
weak references, and a dropped tree closes the file it opened -- unless
another live tree reads the same file.
"""

import gc
import pathlib
import shutil

import ROOT

from grand.dataio import TShower
from grand.dataio.data_tree import grand_tree_list

SAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000" / "shower_1618-13790_L0_0000.root")


def _open_files(path):
    return [f for f in ROOT.gROOT.GetListOfFiles() if f.GetName() == str(path)]


def test_a_dropped_tree_closes_its_file(tmp_path):
    path = tmp_path / "s.root"
    shutil.copy(SAMPLE, path)
    tree = TShower(str(path))
    tree.get_entry(0)
    assert _open_files(path) and tree in grand_tree_list
    del tree
    gc.collect()
    assert not _open_files(path)
    assert not any(t._file is not None and t._file.GetName() == str(path) for t in grand_tree_list)


def test_a_shared_file_stays_open_for_the_other_tree(tmp_path):
    path = tmp_path / "s.root"
    shutil.copy(SAMPLE, path)
    first, second = TShower(str(path)), TShower(str(path))
    first.get_entry(0)
    del first
    gc.collect()
    second.get_entry(1)
    assert int(second.event_number) in (1618, 13790)
    second.stop_using()
    assert not _open_files(path)


def test_stop_using_still_works_and_is_idempotent(tmp_path):
    path = tmp_path / "s.root"
    shutil.copy(SAMPLE, path)
    tree = TShower(str(path))
    tree.stop_using()
    tree.stop_using()
    assert tree not in grand_tree_list and not _open_files(path)
    del tree
    gc.collect()


def test_a_dropped_tree_with_unwritten_entries_is_kept(tmp_path):
    # Dropping it would lose the entries silently: it stays registered until
    # written (an example that let its Event objects go wrote 1 event of 10)
    path = tmp_path / "new.root"
    tree = TShower(str(path))
    tree.run_number, tree.event_number = 1, 5
    tree.fill()
    del tree
    gc.collect()
    (kept,) = [t for t in grand_tree_list if t._file_name == str(path)]
    kept.write()
    kept.stop_using()
    with TShower(str(path)) as reread:
        assert reread.get_number_of_entries() == 1
