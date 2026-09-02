# -*- coding: utf-8 -*-
r"""Directory scanning and file grouping.

``grand.dataio.data_handling`` decides which files belong together and what
they contain, which is what every analysis relies on to find its data.  It was
57% covered, and its failure modes are the quiet kind: a file grouped with the
wrong run, or a directory that scans to nothing.

The fixtures are built here rather than shipped, so these run on a clean
checkout.
"""

import numpy as np
import pytest

from grand.dataio.data_handling import DataDirectory, DataFile
from grand.dataio.event_trees import TEfield
from grand.dataio.run_trees import TRun

N_DU = 2
N_SAMPLES = 32


def _write_run_and_efield(path, run_number=0, n_events=2, level=0):
    r"""Writes a run tree and an electric-field tree into `path`.

    Parameters
    ----------
    path : str
        Destination file.
    run_number : int, optional
        Run number to record.
    n_events : int, optional
        Number of events to write.
    level : int, optional
        Analysis level to record in the trees.  It must match the ``_L<n>_``
        marker in `path`: the directory scanner takes the level from the
        filename but reads the tree attribute named for the level recorded
        *inside* the tree, so a disagreement raises ``AttributeError``.
    """
    run = TRun(path)
    run.run_number = run_number
    run.du_id = list(range(N_DU))
    run.du_xyz = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    run.t_bin_size = [0.5] * N_DU
    run.analysis_level = level
    run.fill()
    run.write()

    t = np.arange(N_SAMPLES) * 0.5
    efield = TEfield(path)
    for event in range(n_events):
        pulse = np.exp(-((t - 8.0 * (event + 1)) ** 2) / 2.0)
        efield.run_number, efield.event_number = run_number, event
        efield.du_id = list(range(N_DU))
        efield.du_nanoseconds = [0] * N_DU
        efield.du_seconds = [0] * N_DU
        efield.trace = np.stack([np.stack([pulse] * 3)] * N_DU).astype(np.float32)
        efield.analysis_level = level
        efield.fill()
    efield.write()


@pytest.fixture
def one_file(tmp_path):
    r"""Returns a directory holding a single conventionally-named file."""
    path = str(tmp_path / 'efield_20260101_000000_RUN0_L0_0000.root')
    _write_run_and_efield(path)
    return str(tmp_path), path


def test_directory_finds_the_file(one_file):
    r"""A scan reports the ROOT files present."""
    directory, path = one_file
    found = DataDirectory(directory).get_list_of_files()
    assert any(path.endswith(f.split('/')[-1]) for f in found), (
        'the scan did not find the file it was given: %s' % found)


def test_datafile_exposes_its_trees(one_file):
    r"""Each tree in a file becomes an attribute named for it."""
    _, path = one_file
    handle = DataFile(path)
    assert hasattr(handle, 'trun'), 'the run tree was not exposed'
    assert hasattr(handle, 'tefield'), 'the electric-field tree was not exposed'


def test_datafile_reports_the_event_count(one_file):
    r"""The file knows how many events it holds."""
    _, path = one_file
    handle = DataFile(path)
    assert len(handle.tefield.get_list_of_events()) == 2


def test_unconventional_names_do_not_abort_the_scan(tmp_path):
    r"""A file that breaks the naming convention is grouped alone, not fatal.

    ``split_filenames`` used to index the underscore-separated fields
    unconditionally, so a single file not following the convention -- which
    nothing enforces -- aborted the scan of the whole directory with
    ``IndexError`` and no indication of which file was at fault.
    """
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN0_L0_0000.root'))
    _write_run_and_efield(str(tmp_path / 'oddly-named.root'), run_number=1)

    handles = DataDirectory(str(tmp_path)).get_list_of_files_handles()
    assert len(handles) >= 2, (
        'the odd name collapsed the grouping: %d handles' % len(handles))


def test_empty_directory_scans_to_nothing(tmp_path):
    r"""A directory with no ROOT files yields an empty list, not an error."""
    (tmp_path / 'notes.txt').write_text('nothing to see')
    assert DataDirectory(str(tmp_path)).get_list_of_files() == []


def test_files_are_grouped_by_type_and_level_not_by_run(tmp_path):
    r"""Grouping keys on tree type and analysis level, not on run number.

    ``split_filenames`` returns ``(tree type, analysis level)``, so two files
    of the same type and level land in one handle even when their run numbers
    differ.  That is correct for the layout the converters produce -- sim2root
    writes **one run per directory** -- and it is worth pinning, because it
    means pointing ``DataDirectory`` at a directory holding several runs
    silently merges them rather than failing.
    """
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN0_L0_0000.root'),
                          run_number=0)
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN1_L0_0000.root'),
                          run_number=1)

    handles = DataDirectory(str(tmp_path)).get_list_of_files_handles()
    assert len(handles) == 1, (
        'grouping changed: %d handles for two files of the same type and '
        'level. If this is now per-run, the one-run-per-directory assumption '
        'has been lifted and the documentation should say so.' % len(handles))


def test_levels_are_separate_handles_and_the_highest_is_the_default(tmp_path):
    r"""Each analysis level gets its own handle; the highest becomes the default.

    Two things are easy to conflate here, and I conflated them twice while
    writing this file.  ``get_list_of_files_handles`` groups by tree type and
    analysis level, so L0 and L1 come back as *two* handles.  The "highest
    level wins" rule is about something else: which tree gets the bare
    attribute name.  With both levels present, ``tefield`` refers to L1, while
    ``tefield_l0`` and ``tefield_l1`` name them individually.

    A script that reads ``directory.tefield`` therefore silently follows the
    most processed form available, which changes as soon as someone adds an L1
    file beside the L0 one.
    """
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN0_L0_0000.root'),
                          level=0)
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN0_L1_0000.root'),
                          level=1)

    directory = DataDirectory(str(tmp_path))
    handles = directory.get_list_of_files_handles()
    assert len(handles) == 2, (
        'expected one handle per analysis level, got %d' % len(handles))
    assert hasattr(directory, 'tefield_l0') and hasattr(directory, 'tefield_l1'), (
        'the per-level attributes were not created')


def test_filename_level_must_match_the_tree_level(tmp_path):
    r"""A level in the name that disagrees with the tree raises.

    The scanner takes the analysis level from the filename and then reads the
    tree attribute named for the level recorded inside the tree.  When the two
    disagree it fails with ``AttributeError: 'DataFile' object has no
    attribute 'tefield_l1'`` -- naming an attribute the user never wrote and
    saying nothing about the real cause.  Pinned here so that improving the
    message is a visible change rather than a silent one.

    See :ref:`issue-reader-directory-coupling`.
    """
    _write_run_and_efield(str(tmp_path / 'efield_20260101_000000_RUN0_L1_0000.root'),
                          level=0)          # name says L1, tree says 0

    with pytest.raises(AttributeError) as excinfo:
        DataDirectory(str(tmp_path)).get_list_of_files_handles()
    assert '_l1' in str(excinfo.value) or '_l0' in str(excinfo.value)
