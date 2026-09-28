# -*- coding: utf-8 -*-
r"""Appending events to a tree must not make ROOT's read cache complain.

grand-mother/grand#89: ``convert_efield2voltage.py`` run over many events
printed, once per event::

    Error in <TTreeCache::FillBuffer>: Inconsistency: fCurrentClusterStart=0
    fEntryCurrent=176 fNextClusterStart=178 but fEntryCurrent should not be in
    between the two

The script writes one event at a time into the same output tree, and grand
reads that tree in full around every append: ``fill_entry_list()`` draws it
when ``TVoltage(filename)`` reopens it, and ``write()`` rebuilds its index.
The TTreeCache keeps the entry window of its previous pass, cut at the number
of entries the tree had then; the tree has since grown past it, and once it
is out of the cache's learning phase (more than 100 entries) and has flushed
a cluster, the next pass lands in a cluster that starts inside the stale
window.  ROOT recovers and the values are right, so it is noise -- but noise
on every event.  Analysing a file on its own did not show it because a
single file rarely held more than 100 events.

The message has been invisible since August 2024 because ``grand.dataio``
sets ``ROOT.gErrorIgnoreLevel = ROOT.kFatal`` on import, so this test lowers
it for its own duration.  ROOT prints straight to file descriptor 2, which
``capfd`` captures.
"""

import numpy as np
import pytest

ROOT = pytest.importorskip("ROOT")

import grand.dataio as groot  # noqa: E402
from grand.dataio.data_tree import DataTree  # noqa: E402

#: Past the TTreeCache learning phase, which is 100 entries.
N_EVENTS = 130

#: A cluster boundary every this many entries, so a small test tree has
#: several; the real output flushes every 30 MB instead.
AUTO_FLUSH = 20

MESSAGE = "TTreeCache::FillBuffer"


@pytest.fixture
def root_errors_shown():
    r"""Lets ROOT print errors for the duration of a test."""
    saved = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    yield
    ROOT.gErrorIgnoreLevel = saved


def _append_events(path, n_events=N_EVENTS):
    r"""Writes events one by one, the way ``Efield2Voltage.save_voltage`` does.

    Parameters
    ----------
    path : pathlib.Path
        Output file; created on the first event.
    n_events : int, optional
        How many events to append.

    Returns
    -------
    list of tuple
        ``(run_number, event_number, trace sum)`` for every entry, read back
        from the file after it is closed.
    """
    fn = str(path)
    rng = np.random.default_rng(0)
    for ev in range(n_events):
        tv = groot.TVoltage(fn)
        if ev == 0:
            tv._tree.SetAutoFlush(AUTO_FLUSH)
        tv.du_count = 1
        tv.run_number = 0
        tv.event_number = ev
        tv.du_id = [ev % 7]
        tv.du_seconds = [1]
        tv.du_nanoseconds = [2]
        tv.trace = rng.normal(size=(1, 3, 16)).astype(np.float32)
        tv.fill()
        tv.write()
    ROOT.gROOT.GetListOfFiles().FindObject(fn).Close()

    f = ROOT.TFile.Open(fn)
    t = f.Get("tvoltage")
    rows = []
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        rows.append((int(t.run_number), int(t.event_number),
                     float(sum(sum(sum(c) for c in du) for du in t.trace))))
    f.Close()
    return rows


def test_appending_many_events_prints_no_cache_error(tmp_path, capfd, root_errors_shown):
    r"""The regression test proper: no TTreeCache message at all."""
    rows = _append_events(tmp_path / "voltage.root")
    err = capfd.readouterr().err
    assert MESSAGE not in err, err[:2000]
    assert [r[:2] for r in rows] == [(0, ev) for ev in range(N_EVENTS)]


def test_the_scenario_does_trigger_the_message_without_the_reset(
        tmp_path, capfd, root_errors_shown, monkeypatch):
    r"""A control: with the cache reset disabled, the same writes do complain.

    Without it the test above could pass because the scenario no longer
    reaches the cache at all.  It also shows the reset changes nothing that
    is read: both runs write identical entries.
    """
    fixed = _append_events(tmp_path / "fixed.root")
    capfd.readouterr()

    monkeypatch.setattr(DataTree, "_reset_read_cache", staticmethod(lambda tree: None))
    unfixed = _append_events(tmp_path / "unfixed.root")
    err = capfd.readouterr().err

    if MESSAGE not in err:
        pytest.skip("this ROOT version no longer reports the stale cache window")
    assert "Inconsistency" in err
    assert unfixed == fixed
