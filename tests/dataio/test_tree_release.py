"""
Releasing tree instances: ``stop_using()`` and the ``with`` form (GitHub issue #71).

Every tree is kept in ``grand_tree_list``, and PyROOT stops owning a TFile once
``TTree.SetDirectory(file)`` is called with it, so before this fix a tree that
opened its file from a name left that file open until the process exited.
Reading many files in a loop grew by about 2.7 MB per file, even with
``stop_using()``.  These tests pin that releasing a tree closes the file it
opened, keeps files other trees or the caller still use, and can be repeated.
"""

import gc
import pathlib
import shutil

import pytest

ROOT = pytest.importorskip("ROOT")

from grand.dataio.data_tree import grand_tree_list, _files_opened_by_trees  # noqa: E402
from grand.dataio.event_trees import TADC  # noqa: E402

ADC_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
    / "adc_1618-13790_L1_0000.root"
)

pytestmark = pytest.mark.skipif(not ADC_FILE.is_file(), reason="committed ADC fixture file is missing")


def _is_open(path):
    """True if ROOT has a file of this name in its list of open files"""
    return bool(ROOT.gROOT.GetListOfFiles().FindObject(str(path)))


def _in_tree_list(tree):
    return any(inst is tree for inst in grand_tree_list)


@pytest.fixture
def adc_copy(tmp_path):
    """A private copy of the committed ADC file, so no other test holds it open"""
    dst = tmp_path / "adc.root"
    shutil.copy(ADC_FILE, dst)
    return dst


def _read_all(tree):
    events = tree.get_list_of_events()
    for event, run in events:
        tree.get_event(event, run)
    return events


def test_with_form_releases_tree_and_closes_file(adc_copy):
    with TADC(str(adc_copy)) as tadc:
        assert _in_tree_list(tadc)
        assert _is_open(adc_copy)
        assert len(_read_all(tadc)) > 0
    assert not _in_tree_list(tadc)
    assert not _is_open(adc_copy)
    assert tadc.tree is None


def test_file_can_be_reopened_after_release(adc_copy):
    with TADC(str(adc_copy)) as tadc:
        first = _read_all(tadc)
    with TADC(str(adc_copy)) as tadc:
        second = _read_all(tadc)
    assert first == second
    assert not _is_open(adc_copy)


def test_with_form_releases_when_the_block_raises(adc_copy):
    with pytest.raises(RuntimeError, match="inside the block"):
        with TADC(str(adc_copy)) as tadc:
            raise RuntimeError("inside the block")
    assert not _in_tree_list(tadc)
    assert not _is_open(adc_copy)


def test_stop_using_twice_is_harmless(adc_copy):
    tadc = TADC(str(adc_copy))
    tadc.stop_using()
    tadc.stop_using()
    assert not _in_tree_list(tadc)
    assert not _is_open(adc_copy)


def test_stop_using_without_closing_keeps_file_open(adc_copy):
    tadc = TADC(str(adc_copy))
    tadc.stop_using(close_file=False)
    assert not _in_tree_list(tadc)
    assert _is_open(adc_copy)
    assert tadc.get_number_of_entries() > 0
    # Now release the file too
    tadc.stop_using()
    assert not _is_open(adc_copy)


def test_shared_file_stays_open_until_last_tree_is_released(adc_copy):
    # The second tree finds the file in gROOT's list of files and reuses it
    first = TADC(str(adc_copy))
    second = TADC(str(adc_copy))
    assert ROOT.addressof(first.file) == ROOT.addressof(second.file)

    first.stop_using()
    assert _is_open(adc_copy)
    assert len(_read_all(second)) > 0

    second.stop_using()
    assert not _is_open(adc_copy)


def test_caller_supplied_tfile_is_left_open(adc_copy):
    f = ROOT.TFile(str(adc_copy), "read")
    try:
        with TADC(f) as tadc:
            assert len(_read_all(tadc)) > 0
        assert f.IsOpen()
        assert ROOT.addressof(f) not in _files_opened_by_trees
    finally:
        f.Close()


def test_loop_over_many_files_leaves_none_open(tmp_path):
    """The issue #71 loop: no ROOT file and no tree survives the iterations"""
    paths = []
    for i in range(10):
        dst = tmp_path / f"adc_{i}.root"
        shutil.copy(ADC_FILE, dst)
        paths.append(dst)

    # Trees other tests dropped are released when the garbage collector runs
    # (#284), which can happen during the loop: collect first, and compare
    # with "no more than" -- the counts fell during the loop on ROOT 6.38
    gc.collect()
    files_before = ROOT.gROOT.GetListOfFiles().GetSize()
    trees_before = len(grand_tree_list)
    for path in paths:
        with TADC(str(path)) as tadc:
            _read_all(tadc)
    assert ROOT.gROOT.GetListOfFiles().GetSize() <= files_before
    assert len(grand_tree_list) <= trees_before
    assert not any(_is_open(p) for p in paths)


def test_memory_growth_per_file_is_bounded(tmp_path):
    """Resident memory stays well below the pre-fix growth of ~2.7 MB per file.

    After the fix the loop grows by ~0.15 MB per file, most of it inside ROOT
    itself.  The bound is 1 MB per file on average, over six times the
    measured growth, so allocator noise cannot trip it.  The files are identical copies
    and are read in a fixed order, so the run is deterministic.
    """
    psutil = pytest.importorskip("psutil")
    n_warmup, n_measured = 5, 30
    paths = []
    for i in range(n_warmup + n_measured):
        dst = tmp_path / f"adc_{i}.root"
        shutil.copy(ADC_FILE, dst)
        paths.append(dst)

    proc = psutil.Process()
    for path in paths[:n_warmup]:
        with TADC(str(path)) as tadc:
            _read_all(tadc)
    rss_start = proc.memory_info().rss
    for path in paths[n_warmup:]:
        with TADC(str(path)) as tadc:
            _read_all(tadc)
    growth_mb_per_file = (proc.memory_info().rss - rss_start) / n_measured / 1e6
    assert growth_mb_per_file < 1.0, f"memory grew by {growth_mb_per_file:.2f} MB per file"
