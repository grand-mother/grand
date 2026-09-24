# -*- coding: utf-8 -*-
r"""Which vertical frame ``xmax_pos_shc`` is written in, measured end to end.

``xmax_pos_shc`` is the position of the shower maximum in shower-core
coordinates: metres from the core, whose origin sits on the ground.  Its ``z``
is therefore a height *above the ground*, not an altitude above sea level.

The distinction is worth a test file because the two differ by the site
altitude -- 1264 m at Xiaodushan -- and because the repository currently holds
one of each:

* ``sim2root/ZHAireSRawRoot/ZHAireSRawToRawROOT.py`` subtracts the ground
  altitude and writes a ground-relative ``z``.  That is correct, and the first
  test measures it against the ZHAireS summary file rather than against the
  converter's own arithmetic.

* The ZHAireS samples committed under ``sim2root/Common/`` do **not** carry
  that subtraction.  They were produced before it existed, so their ``z`` is
  the raw AIRES value, 1264 m too high.

* ``grand/dataio/root_files.py`` compensates for exactly that, subtracting
  ``origin_geoid[2]`` in ``get_simu_parameters`` under the name
  ``FIX_xmax_pos`` (the "DC2 FIX", collab-issues#34).

Those last two cancel, which is why the shipped samples read correctly today.
They cancel *only* for data of that vintage.  Regenerate the samples with the
current converter -- an ordinary, unremarkable thing to do -- and the
compensation is applied to data that no longer needs it, putting Xmax 1264 m
**below** where it belongs, silently.  The last test states that trap so that
whoever regenerates the samples meets it as a failure rather than as a subtly
wrong reconstruction months later.

These tests run the real converters on the repository's own fixtures.  Nothing
here reasons from the source text: the units question was settled once by
reading a diff and the reading was wrong in both directions.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

ZHAIRES_CONVERTER = ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'ZHAireSRawToRawROOT.py'
COREAS_CONVERTER = ROOT / 'sim2root' / 'CoREASRawRoot' / 'CoreasToRawROOT.py'
COREAS_FIXTURE = ROOT / 'sim2root' / 'CoREASRawRoot' / 'proton'

#: The two committed ZHAireS events, and the sample directory built from them.
ZHAIRES_EVENTS = {
    1618: ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618',
    13790: ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'GP300_Xi_Sib_Proton_3.87_79.4_310.0_13790',
}
SAMPLE = (ROOT / 'sim2root' / 'Common'
          / 'sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000')


def _have(module):
    r"""Returns whether ``module`` can be imported.

    Parameters
    ----------
    module : str
        The module name.

    Returns
    -------
    bool
        True if the import succeeds.
    """
    try:
        __import__(module)
    except ImportError:
        return False
    return True


#: The converters build ROOT trees, so PyROOT has to be present.  uproot reads
#: the results back; it is a declared dependency but checked all the same, so
#: a partial environment skips rather than errors.
needs_root = pytest.mark.skipif(
    not (_have('ROOT') and _have('uproot')),
    reason='PyROOT and uproot are both needed to run a converter and read it back')


def _sry_truth(folder):
    r"""Returns Xmax's raw ``z`` and the ground altitude, read from the summary.

    This is the independent oracle: the numbers come from the ZHAireS ``.sry``
    text, not from anything the converter computed.

    Parameters
    ----------
    folder : pathlib.Path
        A ZHAireS output directory holding exactly one ``.sry``.

    Returns
    -------
    tuple of float
        ``(xmax_z_m, ground_altitude_m)``, both in metres.
    """
    sry = sorted(folder.glob('*.sry'))
    assert len(sry) == 1, 'expected one .sry in %s, found %d' % (folder, len(sry))
    text = sry[0].read_text(encoding='utf-8', errors='replace')

    # "                     Ground altitude: 1.2640 km (904.5084 g/cm2)"
    ground = re.search(r'Ground altitude:\s+([0-9.eE+-]+)\s+km', text)
    assert ground, 'no "Ground altitude" line in %s' % sry[0]

    # "  Pos. Max.:     5.76594    7.25009   -4.05065    3.98891    5.76341"
    #                  altitude   distance   x          y          z
    pos = re.search(r'Pos\. Max\.:\s+' + r'\s+'.join([r'(-?[0-9.eE+-]+)'] * 5), text)
    assert pos, 'no "Pos. Max." line in %s' % sry[0]

    return float(pos.group(5)) * 1000.0, float(ground.group(1)) * 1000.0


def _convert_zhaires(folder, event_id, workdir):
    r"""Runs the ZHAireS converter and returns the raw shower tree's fields.

    Parameters
    ----------
    folder : pathlib.Path
        The ZHAireS output directory to convert.
    event_id : int
        The event number to record.
    workdir : pathlib.Path
        Directory to run in and write the output into.

    Returns
    -------
    dict
        ``{branch_name: numpy array}`` for the produced ``trawshower``.
    """
    import uproot

    out = 'raw_%d.root' % event_id
    completed = subprocess.run(
        [sys.executable, str(ZHAIRES_CONVERTER), str(folder),
         'standard', '1', str(event_id), out],
        cwd=str(workdir), env={**_env()}, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, (
        'the ZHAireS converter failed on %s:\n%s\n%s'
        % (folder.name, completed.stdout[-2000:], completed.stderr[-2000:]))

    produced = workdir / out
    assert produced.is_file(), 'the converter reported success but wrote nothing'
    tree = uproot.open(str(produced))['trawshower']
    return {name: tree[name].array(library='np') for name in tree.keys()}


def _env():
    r"""Returns an environment with the repository importable.

    Returns
    -------
    dict
        A copy of the current environment with ``PYTHONPATH`` set.
    """
    import os

    env = dict(os.environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + existing if existing else '')
    return env


@needs_root
@pytest.mark.parametrize('event_id', sorted(ZHAIRES_EVENTS))
def test_the_zhaires_converter_writes_xmax_above_the_ground(event_id, tmp_path):
    r"""``xmax_pos_shc[2]`` is the AIRES ``z`` minus the ground altitude.

    Measured against the ``.sry``, which is the simulation's own record and
    owes nothing to the converter.  AIRES reports Xmax's ``z`` from sea level;
    the shower-core frame counts from the ground, so the ground altitude comes
    off.  At Xiaodushan that is 1264 m.

    Both committed events are checked.  13790 matters as much as 1618: at a
    zenith of 79.4 degrees Xmax is 69 km away and Earth curvature puts 361 m
    between ``xmax_alt - site_alt`` and the true ``z``, so a test written
    against altitudes rather than against the ``.sry`` would be wrong about
    which frame it was even testing.
    """
    folder = ZHAIRES_EVENTS[event_id]
    if not folder.is_dir():
        pytest.skip('the ZHAireS fixture %s is not present' % folder.name)

    raw_z, ground = _sry_truth(folder)
    fields = _convert_zhaires(folder, event_id, tmp_path)
    written_z = float(fields['xmax_pos_shc'][0][2])

    assert written_z == pytest.approx(raw_z - ground, abs=0.5), (
        'xmax_pos_shc[2] is %.2f m; the .sry puts Xmax at z=%.2f m above sea '
        'level with the ground at %.2f m, so the shower-core frame value is '
        '%.2f m. A difference of exactly the ground altitude means the '
        'subtraction in ZHAireSRawToRawROOT.py has been lost.'
        % (written_z, raw_z, ground, raw_z - ground))

    assert abs(written_z - raw_z) > ground / 2.0, (
        'xmax_pos_shc[2] equals the sea-level value, so the frame is wrong '
        'even though the arithmetic above happened to pass')


@needs_root
def test_the_committed_zhaires_sample_predates_that_subtraction(tmp_path):
    r"""The shipped sample's Xmax is the sea-level value, 1264 m too high.

    Not a defect in today's converter -- the test above shows it writes the
    ground-relative value -- but a property of the committed artefact, which
    was produced before the subtraction existed.  It is pinned because the
    "DC2 FIX" in ``grand/dataio/root_files.py`` is calibrated to it, so the
    two are a matched pair and neither can be changed alone.
    """
    import uproot

    shower = SAMPLE / 'shower_1618-13790_L0_0000.root'
    run = SAMPLE / 'run_1_L0_0000.root'
    if not shower.is_file() or not run.is_file():
        pytest.skip('the committed ZHAireS sample is not present')

    tree = uproot.open(str(shower))['tshower']
    events = list(tree['event_number'].array(library='np'))
    shipped = tree['xmax_pos_shc'].array(library='np')[events.index(1618)]
    geoid = uproot.open(str(run))['trun']['origin_geoid'].array(library='np')[0]

    raw_z, ground = _sry_truth(ZHAIRES_EVENTS[1618])
    assert float(geoid[2]) == pytest.approx(ground, abs=1.0), (
        'origin_geoid[2] and the .sry ground altitude have diverged; the '
        'cancellation this file describes no longer holds as described')
    assert float(shipped[2]) == pytest.approx(raw_z, abs=1.0), (
        'the committed sample no longer carries the sea-level Xmax. If it '
        'was regenerated with the current converter, the DC2 FIX in '
        'grand/dataio/root_files.py now over-corrects it -- see the next test')


@needs_root
def test_the_dc2_xmax_fix_would_over_correct_regenerated_data(tmp_path):
    r"""The DC2 FIX is calibrated to the stale sample, not to the converter.

    ``get_simu_parameters`` computes::

        FIX_xmax_pos = xmax_pos_shc + shower_core_pos - [0, 0, origin_geoid[2]]

    Against the committed sample that lands on the right answer, because the
    sample's ``xmax_pos_shc`` is high by exactly ``origin_geoid[2]``.  Against
    data from today's converter the input is already correct and the same
    subtraction moves Xmax 1264 m underground.

    This is the failure mode worth guarding: it needs no code change to
    appear.  Someone regenerates the samples -- the natural response to the
    test above -- and every consumer of ``FIX_xmax_pos`` silently shifts.
    """
    folder = ZHAIRES_EVENTS[1618]
    if not folder.is_dir():
        pytest.skip('the ZHAireS fixture is not present')

    raw_z, ground = _sry_truth(folder)
    fields = _convert_zhaires(folder, 1618, tmp_path)

    fresh_z = float(fields['xmax_pos_shc'][0][2])
    core_z = float(fields['shower_core_pos'][0][2])

    # What root_files.py would hand a consumer for this event.
    fixed_z = fresh_z + core_z - ground

    assert fixed_z == pytest.approx(raw_z - ground - ground, abs=1.0), (
        'the DC2 FIX no longer double-subtracts the site altitude for freshly '
        'converted data. If root_files.py learned to tell the two vintages '
        'apart, this test has done its job and should be replaced by one '
        'asserting the corrected behaviour')

    assert fixed_z < 0.0 or fixed_z < (raw_z - ground) - ground / 2.0, (
        'expected the double subtraction to be plainly visible')


@needs_root
def test_the_coreas_converter_completes_on_its_own_fixture(tmp_path):
    r"""``CoreasToRawROOT.py`` converts ``proton/`` without raising.

    It did not, until the ``Xmax_NWU`` branch was repaired: the name is
    computed only when the ``.reas`` carries ``ShowerZenithAngle``, and this
    fixture's ``.reas`` does not, so the unconditional write of
    ``RawShower.xmax_pos_shc`` raised ``UnboundLocalError`` partway through --
    after the output file had been created and partly filled.  See issue #159.

    The fixture is the only CoREAS input in the repository, so this is also
    the only end-to-end coverage the CoREAS path has.
    """
    import uproot
    import numpy as np

    if not COREAS_FIXTURE.is_dir():
        pytest.skip('the CoREAS fixture is not present')

    completed = subprocess.run(
        [sys.executable, str(COREAS_CONVERTER), '-d', str(COREAS_FIXTURE)],
        cwd=str(tmp_path), env=_env(), capture_output=True, text=True, timeout=1800)

    assert completed.returncode == 0, (
        'the CoREAS converter failed on its own fixture:\n%s\n%s'
        % (completed.stdout[-2000:], completed.stderr[-3000:]))

    produced = sorted(tmp_path.glob('Coreas_*.rawroot'))
    assert produced, 'the converter reported success but wrote no .rawroot'

    tree = uproot.open(str(produced[0]))['trawshower']

    # This path has no DistanceOfShowerMaximum, so Xmax's position is unknown.
    # It must not be recorded as a usable number: zeros would read as a real
    # Xmax at the array origin.
    xmax = tree['xmax_pos_shc'].array(library='np')[0]
    assert np.all(np.isnan(np.asarray(xmax, dtype=float))), (
        'xmax_pos_shc is %r; this branch cannot know Xmax, so it has to say '
        'so rather than write a position that looks real' % (xmax,))

    # The site altitude reaches origin_geoid.  The hard-coded site table is in
    # centimetres and the observation level is in metres; if the override that
    # keeps them apart is ever removed, this is where it shows up.
    site_alt = float(tree['site_alt'].array(library='np')[0])
    assert 0.0 < site_alt < 10000.0, (
        'site_alt is %.1f, which is not a plausible altitude in metres -- the '
        'centimetre site table may have reached it' % site_alt)
