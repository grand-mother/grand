# -*- coding: utf-8 -*-
r"""Covers the file readers in :mod:`grand.dataio.root_files`.

``root_files.py`` is the layer an analysis actually opens a file with, and only
32 % of it was covered -- 118 of 174 statements.  The obstacle was not the
reader but the fixture: it cannot be handed a file.

**What the reader requires, which is written down nowhere else.**
:class:`~grand.dataio.root_files.FileEfield` does not read the file it is
given.  It reads the *directory* the file is in, through
:class:`~grand.dataio.data_handling.DataDirectory`, and looks up the run and
shower trees there.  ``DataDirectory`` in turn groups files by **filename
prefix** -- ``ftshowers`` collects files whose name begins with ``shower_`` --
so it does not matter which trees a file actually contains.  A single file
holding ``TRun``, ``TEfield`` and ``TShower`` is only ever seen as an efield
file, and the reader then fails with

.. code-block:: text

    AttributeError: 'NoneType' object has no attribute 'file_name'

Three conditions have to hold together, and the fixture below satisfies all
three:

1. the run, efield and shower trees live in **separate files**, named
   ``run_*``, ``efield_*`` and ``shower_*``;
2. each name carries its analysis level as ``_L0_`` or ``_L1_``;
3. the ``analysis_level`` stored *in each tree* matches the name.

See ``issue-reader-directory-coupling`` in the documentation.
"""

import os

import numpy as np
import pytest

from grand.dataio.event_trees import TEfield, TShower
from grand.dataio.root_files import FileEfield
from grand.dataio.run_trees import TRun

N_DU, N_SAMPLES, N_EVENTS = 3, 128, 2
T_BIN_NS = 0.5
SITE = [40.98, 93.95, 1200.0]

#: The rest of a GRAND file name after the ``run_``/``efield_``/``shower_``
#: prefix: date, time, run, analysis level, sub-run.
STEM = '20260101_000000_RUN0_L0_0000.root'


@pytest.fixture(scope='module')
def event_dir(tmp_path_factory):
    r"""Returns a directory holding a readable three-file event set.

    Built from the tree classes rather than shipped, for the reason given in
    ``tests/sim/test_pipeline_end_to_end.py``: ``data/`` is gitignored, so a
    checked-in ROOT file cannot be read in CI.

    Returns
    -------
    tuple of str
        The directory and the path of the efield file within it.
    """
    directory = tmp_path_factory.mktemp('reader')
    paths = {kind: str(directory / ('%s_%s' % (kind, STEM)))
             for kind in ('run', 'efield', 'shower')}

    run = TRun(paths['run'])
    run.run_number, run.analysis_level = 0, 0
    run.du_id = list(range(N_DU))
    run.du_xyz = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]
    run.t_bin_size = [T_BIN_NS] * N_DU
    run.origin_geoid = SITE
    run.fill()
    run.write()

    for event in range(N_EVENTS):
        # A different amplitude per event and per arm, so that loading the
        # wrong event or transposing an axis changes the answer rather than
        # producing something symmetric.
        rng = np.random.default_rng(event)
        trace = rng.normal(0.0, 1.0, (N_DU, 3, N_SAMPLES)).astype(np.float32)
        trace *= np.array([1.0, 0.5, 0.25])[None, :, None] * (event + 1)

        efield = TEfield(paths['efield'])
        efield.run_number, efield.event_number = 0, event
        efield.analysis_level = 0
        efield.du_id = list(range(N_DU))
        efield.du_seconds = [0] * N_DU
        efield.du_nanoseconds = [100 * event] * N_DU
        efield.trace = trace
        efield.fill()
        efield.write()

        shower = TShower(paths['shower'])
        shower.run_number, shower.event_number = 0, event
        shower.analysis_level = 0
        shower.zenith, shower.azimuth = 80.0 + event, 10.0 * event
        shower.energy_primary = 1e9
        shower.shower_core_pos = [0.0, 0.0, 1200.0]
        shower.xmax_pos_shc = [0.0, 0.0, 10000.0]
        shower.fill()
        shower.write()

    return str(directory), paths['efield']


@pytest.fixture(scope='module')
def reader(event_dir):
    r"""Returns a :class:`FileEfield` open on the fixture."""
    return FileEfield(event_dir[1])


def test_the_reader_opens_a_three_file_event_set(reader):
    r"""Construction succeeds and finds both events."""
    assert reader.get_nb_events() == N_EVENTS, (
        'expected %d events, got %d' % (N_EVENTS, reader.get_nb_events()))
    assert reader.tt_run is not None, 'the run tree was not located'
    assert reader.tt_shower is not None, 'the shower tree was not located'


def test_trace_geometry_matches_what_was_written(reader):
    r"""Sample count and sampling rate come back as written.

    ``t_bin_size`` is stored in nanoseconds and the reader reports megahertz,
    so 0.5 ns must become 2000 MHz.  A factor of a thousand here would be easy
    to miss and would rescale every spectrum downstream.
    """
    assert reader.get_size_trace() == N_SAMPLES
    freqs = np.atleast_1d(reader.get_sampling_freq_mhz())
    assert np.allclose(freqs, 2000.0), (
        'sampling frequency came back as %s MHz, not 2000 for a 0.5 ns bin'
        % freqs)


def test_loading_by_index_gives_the_right_event(reader):
    r"""Each index loads its own traces, and they differ between events.

    The fixture scales event 1 by two, so identical results would mean the
    index was ignored -- which is exactly the failure a reader that caches too
    eagerly would produce.
    """
    reader.load_event_idx(0)
    first = np.array(reader.get_obj_handling3dtraces().traces, copy=True)
    reader.load_event_idx(1)
    second = np.array(reader.get_obj_handling3dtraces().traces, copy=True)

    assert first.shape == (N_DU, 3, N_SAMPLES), 'unexpected shape %s' % (first.shape,)
    assert not np.allclose(first, second), 'both indices gave the same traces'
    assert np.isclose(np.abs(second).max() / np.abs(first).max(), 2.0, rtol=0.3), (
        'event 1 should be about twice event 0; ratio was %.3f'
        % (np.abs(second).max() / np.abs(first).max()))


def test_sequential_loading_walks_the_file(reader):
    r"""``load_next_event`` advances rather than repeating."""
    reader.load_event_idx(0)
    seen = [np.array(reader.get_obj_handling3dtraces().traces, copy=True)]
    reader.load_next_event()
    seen.append(np.array(reader.get_obj_handling3dtraces().traces, copy=True))
    assert not np.allclose(seen[0], seen[1]), 'load_next_event did not advance'


def test_the_traces_object_is_tagged_as_an_efield(reader):
    r"""``FileEfield`` labels what it returns, which is how units follow the data."""
    reader.load_event_idx(0)
    traces = reader.get_obj_handling3dtraces()
    assert traces.type_trace == 'Efield', (
        'traces came back tagged %r' % traces.type_trace)
    assert np.isfinite(np.asarray(traces.traces)).all()


def test_du_trigger_times_are_returned_with_their_origin(reader):
    r"""``get_du_nanosec_ordered`` returns a ``(times, origin)`` pair.

    Not an array, which is what its docstring claimed until this test was
    written: ``np.asarray`` on the result raises
    ``ValueError: setting an array element with a sequence``.  The second
    counts are reduced to their minimum and the difference folded into the
    nanoseconds, so the times are comparable within an event.

    Despite the name, the values are in unit order and are not sorted.
    """
    reader.load_event_idx(0)
    result = reader.get_du_nanosec_ordered()

    assert isinstance(result, tuple) and len(result) == 2, (
        'expected a (times, origin) pair, got %r' % (type(result),))
    times, origin = result
    times = np.asarray(times)
    assert times.shape == (N_DU,), (
        'expected one time per unit, got shape %s' % (times.shape,))
    assert np.isfinite(times).all()
    assert np.isfinite(origin), 'the second origin is not finite'
    assert times.min() >= 0.0, 'times are measured from the earliest second'


@pytest.mark.xfail(reason='get_du_count() returns 0 on a file whose TRun was '
                          'written with du_id set; the count appears to be read '
                          'from a field the tree classes do not populate. The '
                          'traces themselves carry the right number of units.',
                   strict=False)
def test_du_count_matches_the_traces(reader):
    r"""``get_du_count`` should agree with the first axis of the traces.

    It does not: the reader reports 0 while the traces are ``(3, 3, 128)``.
    Recorded rather than asserted away, because a caller sizing an array from
    ``get_du_count()`` would silently get nothing.
    """
    reader.load_event_idx(0)
    traces = np.asarray(reader.get_obj_handling3dtraces().traces)
    assert reader.get_du_count() == traces.shape[0]


def test_a_filename_without_an_analysis_level_is_refused(tmp_path):
    r"""A name carrying neither ``_L0_`` nor ``_L1_`` raises, and says why.

    A bare ``raise`` used to stand here with no exception active, so such a
    file failed with "No active exception to reraise" -- naming neither the
    file nor the convention it broke.
    """
    path = str(tmp_path / 'oddly-named.root')
    run = TRun(path)
    run.run_number, run.analysis_level = 0, 0
    run.du_id = [0]
    run.du_xyz = [[0.0, 0.0, 0.0]]
    run.t_bin_size = [T_BIN_NS]
    run.origin_geoid = SITE
    run.fill()
    run.write()

    efield = TEfield(path)
    efield.run_number, efield.event_number, efield.analysis_level = 0, 0, 0
    efield.du_id = [0]
    efield.du_seconds, efield.du_nanoseconds = [0], [0]
    efield.trace = np.zeros((1, 3, N_SAMPLES), dtype=np.float32)
    efield.fill()
    efield.write()

    with pytest.raises(ValueError, match='_L0_'):
        FileEfield(path)


def test_the_reader_needs_the_trees_in_separately_named_files(tmp_path):
    r"""All three trees in one file is not readable, and that is the trap.

    ``DataDirectory`` groups by filename prefix, so a file called ``efield_*``
    is never searched for a shower tree however many it contains.  Asserted so
    that the day the reader looks inside the file it was handed, this fails and
    the caveat can be removed from the documentation.
    """
    path = str(tmp_path / ('efield_%s' % STEM))

    run = TRun(path)
    run.run_number, run.analysis_level = 0, 0
    run.du_id = list(range(N_DU))
    run.du_xyz = [[0.0, 0.0, 0.0]] * N_DU
    run.t_bin_size = [T_BIN_NS] * N_DU
    run.origin_geoid = SITE
    run.fill()
    run.write()

    efield = TEfield(path)
    efield.run_number, efield.event_number, efield.analysis_level = 0, 0, 0
    efield.du_id = list(range(N_DU))
    efield.du_seconds = [0] * N_DU
    efield.du_nanoseconds = [0] * N_DU
    efield.trace = np.zeros((N_DU, 3, N_SAMPLES), dtype=np.float32)
    efield.fill()
    efield.write()

    shower = TShower(path)
    shower.run_number, shower.event_number, shower.analysis_level = 0, 0, 0
    shower.zenith, shower.azimuth = 80.0, 0.0
    shower.energy_primary = 1e9
    shower.shower_core_pos = [0.0, 0.0, 1200.0]
    shower.xmax_pos_shc = [0.0, 0.0, 10000.0]
    shower.fill()
    shower.write()

    assert os.path.exists(path)
    with pytest.raises(AttributeError):
        FileEfield(path)
