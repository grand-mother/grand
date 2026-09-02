# -*- coding: utf-8 -*-
r"""The current reader still reads the ROOT files of 2024.

GRAND files outlive the code that wrote them.  Simulation campaigns are
expensive, the collaboration keeps their output, and a reader that quietly
stops handling last year's files invalidates work nobody is going to redo.
Nothing tested that: the tree tests all write a file and read it back with the
same code in the same process, which cannot fail the way real reading fails.

The fixtures used here are the ones already in the repository, committed in
April and October 2024 by the sim2root converters of that time -- a genuine
two-year gap, and no one has to maintain them.  ``TADC`` has gained 24 fields
since, so this also pins what happens when the code asks a file for a branch
it does not have.

The tests assert the *values*, not just that reading did not raise.  A reader
that silently returned defaults for everything would pass a smoke test and
would have lost the data.
"""

import pathlib

import numpy as np
import pytest

import grand.dataio as groot

#: The 2024 fixtures.  Committed with the repository, so a fresh clone has
#: them; the skip below is for a partial checkout rather than for a normal run.
FIXTURES = pathlib.Path(__file__).resolve().parents[2] / 'sim2root' / 'Common'

#: CoREAS, Dunhuang, committed 2024-04-04.
DUNHUANG = FIXTURES / 'sim_Dunhuang_20170401_000000_RUN1_CD_CoREAS-NJ_0000'

#: ZHAireS, Xiaodushan, committed 2024-10-03.
XIAODUSHAN = FIXTURES / 'sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS_0000'

#: Fields added to ``TADC`` after the Dunhuang fixture was written.  Reading
#: one of these from that file is the interesting case: the branch does not
#: exist, and the question is whether the reader says so, raises, or invents
#: something.
FIELDS_ADDED_SINCE_2024 = ['nutrig_rhox', 'nutrig_rhoy', 'gps_receiver_mode',
                           'adc_temp', 'hardware_id', 'data_format_version',
                           'trigger_status', 'pps_id']


def _open(cls, path):
    r"""Returns a tree of class `cls` reading `path`, positioned on entry 0.

    Parameters
    ----------
    cls : type
        Tree class, for example ``groot.TRun``.
    path : pathlib.Path
        The ROOT file.

    Returns
    -------
    object
        The tree, or a skip if the fixture is not in this checkout.
    """
    if not path.exists():
        pytest.skip('fixture %s is not in this checkout' % path.name)
    tree = cls(_file_name=str(path))
    tree.get_entry(0)
    return tree


def test_a_2024_run_tree_still_reads():
    r"""``TRun`` written in April 2024 gives back its run metadata.

    Site name, array origin, per-unit identifiers and sampling interval:
    everything a later analysis needs in order to interpret the event trees
    that go with it.
    """
    run = _open(groot.TRun, DUNHUANG / 'run_1_L0_0000.root')

    assert run.run_number == 1
    assert run.site == 'Dunhuang'
    assert len(list(run.du_id)) == 289, 'lost the detector-unit list'
    assert list(run.du_id)[:4] == [1, 2, 3, 4]
    assert list(run.t_bin_size)[:2] == [1.0, 1.0], 'sampling interval, in ns'

    latitude, longitude, _ = list(run.origin_geoid)
    assert latitude == pytest.approx(40.1421, abs=1e-3)
    assert longitude == pytest.approx(94.6619, abs=1e-3)


def test_a_2024_adc_tree_still_reads():
    r"""``TADC`` written in April 2024 gives back its traces, at full shape.

    289 units by 3 channels by 2048 samples.  The shape matters as much as
    the numbers: a nested vector read back flattened would still be 1.8
    million values, all of them present and none of them attributable to a
    detector unit.
    """
    adc = _open(groot.TADC, DUNHUANG / 'adc_4100-4100_L1_0000.root')

    assert adc.run_number == 1
    assert adc.event_number == 4100
    assert adc.du_count == 289

    traces = adc.trace_ch
    assert len(traces) == 289, 'one entry per detector unit'
    assert len(traces[0]) == 3, 'three channels'
    assert len(traces[0][0]) == 2048, 'trace length'


def test_a_2024_shower_tree_still_reads():
    r"""``TShower`` written in April 2024 gives back the shower parameters."""
    shower = _open(groot.TShower, DUNHUANG / 'shower_4100-4100_L0_0000.root')

    assert shower.energy_primary == pytest.approx(1e8), 'primary energy, GeV'
    assert shower.zenith == pytest.approx(55.0), 'degrees'
    assert shower.azimuth == pytest.approx(-13.57, abs=1e-2), 'degrees'


def test_a_2024_efield_tree_still_reads():
    r"""``TEfield`` written in April 2024 gives back traces of the right shape."""
    efield = _open(groot.TEfield, DUNHUANG / 'efield_4100-4100_L1_0000.root')

    assert len(list(efield.du_id)) > 0, 'no detector units in the event'
    assert len(efield.trace) == len(list(efield.du_id)), (
        'one trace per detector unit')
    assert len(efield.trace[0]) == 3, 'three field components'


@pytest.mark.parametrize('field', FIELDS_ADDED_SINCE_2024)
def test_fields_added_since_2024_read_as_empty(field):
    r"""A branch the file does not have reads as empty, not as an error.

    The behaviour that makes the schema extensible: 24 fields were added to
    ``TADC`` after this file was written, and asking an old file for one of
    them yields an empty vector rather than raising or returning a value that
    was never measured.

    Empty is the honest answer -- "this file does not carry that" -- and it
    is distinguishable from a real reading of zero, which would come back as
    a vector of zeros with one element per detector unit.
    """
    adc = _open(groot.TADC, DUNHUANG / 'adc_4100-4100_L1_0000.root')

    value = getattr(adc, field)
    assert len(list(value)) == 0, (
        '%s is absent from the 2024 file but read back as %r; a value that '
        'was never written is worse than no value' % (field, list(value)))


def test_the_zhaires_fixtures_still_read():
    r"""The other converter's output of October 2024 also still reads.

    Two converters write these files and they do not agree about everything
    (see the next test), so both paths are exercised.
    """
    run = _open(groot.TRun, XIAODUSHAN / 'run_0_L0_0000.root')

    assert run.site == 'Xiaodushan'
    assert run.run_number == 0
    latitude, longitude, altitude = list(run.origin_geoid)
    assert latitude == pytest.approx(40.99, abs=1e-2)
    assert longitude == pytest.approx(93.94, abs=1e-2)
    assert altitude == pytest.approx(1264.0, abs=1.0), 'metres above the geoid'


def test_the_two_2024_fixtures_disagree_about_the_altitude_unit():
    r"""One fixture's array origin is in metres and the other's is in centimetres.

    The ZHAireS fixture puts Xiaodushan at 1264, and the CoREAS fixture puts
    Dunhuang at 114200 -- which is 1142 m, the actual altitude of the site,
    expressed in centimetres.

    The cause is known and is not in today's converter.  Until commit
    ``0694fa9`` (2024-11-04, "save obs level as site altitude"), the CoREAS
    path wrote the altitude straight from a hard-coded site table in
    ``sim2root/CoREASRawRoot/CorsikaInfoFuncs.py``, whose entries are in
    centimetres and say so in a comment.  That commit made the converter
    overwrite the value with the observation level in metres three lines
    later, so files written since are in metres.  The April 2024 fixture
    predates it.

    This test exists so that the discrepancy is a stated, explained property
    of the fixtures rather than a surprise to whoever next reads them
    together -- and so that it is noticed if someone "fixes" the fixture
    without knowing why it is like that. The unit itself is not recorded
    anywhere in the schema, which is the underlying problem; see the known
    issues.
    """
    dunhuang = _open(groot.TRun, DUNHUANG / 'run_1_L0_0000.root')
    xiaodushan = _open(groot.TRun, XIAODUSHAN / 'run_0_L0_0000.root')

    assert list(dunhuang.origin_geoid)[2] == pytest.approx(114200.0, abs=1.0)
    assert list(xiaodushan.origin_geoid)[2] == pytest.approx(1264.0, abs=1.0)

    ratio = list(dunhuang.origin_geoid)[2] / 1142.0
    assert ratio == pytest.approx(100.0, abs=0.5), (
        'the Dunhuang value is no longer 100x the site altitude in metres, '
        'so the centimetre explanation above may no longer be the right one')


def test_compatibility_comes_from_branch_names_and_not_from_a_version():
    r"""The file carries a version field, and nothing reads it.

    Worth stating because it is the property everything above depends on.
    ``TADC`` and ``TRun`` both declare ``event_version``, documented as "Event
    format version of the DAQ", and the 2024 file carries it -- holding 0.
    But no code in :mod:`grand` ever reads it: the only mentions outside the
    two declarations are in ``tests/dataio/test_root_trees.py``, which sets
    it, and two commented-out lines in ``granddb/rootdblib.py``.

    So nothing negotiates a schema version. The reader binds branches by name
    and type, and a file stays readable exactly as long as the names in it
    still mean what they meant. That is why adding a field is safe -- the 24
    fields tested above -- and why *renaming* one is not: the old branch
    becomes unreachable and the new one reads empty, with no error anywhere.
    ``tests/dataio/test_schema_snapshot.py`` is what makes a rename visible
    in review, since nothing at runtime will.

    If the reader ever does start consulting ``event_version``, this test
    fails and the reasoning above needs redoing.
    """
    adc = _open(groot.TADC, DUNHUANG / 'adc_4100-4100_L1_0000.root')

    branches = {branch.GetName() for branch in adc._tree.GetListOfBranches()}
    assert 'event_version' in branches, (
        'the 2024 file no longer carries the DAQ format version')
    assert adc.event_version == 0, (
        'the DAQ format version is %r, not the 0 every 2024 file carries'
        % adc.event_version)

    # Every mention of the name inside the package must be one of the two
    # declarations.  Anything else is code that consults it.
    package = pathlib.Path(__file__).resolve().parents[2] / 'grand'
    mentions = []
    for path in sorted(package.rglob('*.py')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if 'event_version' in line:
                mentions.append('%s:%d: %s'
                                % (path.name, number, line.strip()))

    assert len(mentions) == 2, (
        'expected event_version to appear twice in grand/, as the TADC and '
        'TRun declarations; found %d:\n  %s'
        % (len(mentions), '\n  '.join(mentions)))
    assert all('TTreeScalarDesc' in mention for mention in mentions), (
        'grand/ now does something with event_version other than declare '
        'it:\n  %s' % '\n  '.join(mentions))

    assert np.asarray(adc.trace_ch[0][0]).size == 2048, (
        'and the data is readable without any version having been consulted')
