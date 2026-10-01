# -*- coding: utf-8 -*-
r"""A shower that hit no antenna goes through the whole chain (issue #91).

Effective-area studies need every simulated shower, including the ones that
missed the array: dropping them biases the acceptance.  So the policy is that
such a shower is *kept*, with its shower and run information, ``du_count`` 0
and empty per-antenna arrays, at every level:

* ``ZHAireSRawToRawROOT.py`` writes it to the rawroot file;
* ``sim2root.py`` writes it to the shower, showersim and efield files (it used
  to die in ``get_tree_du_id_xyz_geoid`` with ``ValueError: ecef coordinates
  must be n x 3``);
* ``Efield2Voltage`` writes it to the voltage file (it used to raise
  ``IndexError`` on ``f_samp_mhz[0]``);
* ``scripts/convert_voltage2adc.py`` writes it to the ADC file (same
  ``IndexError``);
* the ``grand.dataio.root_files`` readers load it; the one method that cannot
  do anything with zero antennas, ``get_obj_handling3dtraces``, says so.

The input is the committed ZHAireS fixture with its antenna files removed,
which is exactly what ZHAireS leaves when no antenna was selected.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

ZHAIRES_CONVERTER = ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'ZHAireSRawToRawROOT.py'
SIM2ROOT = ROOT / 'sim2root' / 'Common' / 'sim2root.py'
VOLTAGE2ADC = ROOT / 'scripts' / 'convert_voltage2adc.py'
FIXTURE = ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618'
EVENT_ID = 1618


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


pytestmark = [
    pytest.mark.skipif(not (_have('ROOT') and _have('uproot')),
                       reason='PyROOT and uproot are both needed'),
    pytest.mark.skipif(not FIXTURE.is_dir(), reason='the ZHAireS fixture is not present'),
]


def _env():
    r"""Returns an environment with the repository importable.

    Returns
    -------
    dict
        A copy of the current environment with ``PYTHONPATH`` set.
    """
    env = dict(os.environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + existing if existing else '')
    return env


def _run(args, cwd):
    r"""Runs a script with the current interpreter and fails on a non-zero exit.

    Parameters
    ----------
    args : list of str
        The script and its arguments.
    cwd : pathlib.Path
        Directory to run in.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with its output captured as text.
    """
    completed = subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(cwd),
                               env=_env(), capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, (
        '%s failed:\n%s\n%s' % (pathlib.Path(str(args[0])).name,
                                completed.stdout[-3000:], completed.stderr[-3000:]))
    return completed


def _convert(folder, out):
    r"""Runs the ZHAireS converter on `folder` and returns the rawroot path.

    Parameters
    ----------
    folder : pathlib.Path
        ZHAireS output directory.
    out : pathlib.Path
        The rawroot file to write.

    Returns
    -------
    pathlib.Path
        `out`, once written.
    """
    _run([ZHAIRES_CONVERTER, folder, 'standard', '1', str(EVENT_ID), out.name], out.parent)
    assert out.is_file(), 'the converter reported success but wrote nothing'
    return out


def _sim2root(raws, out, *extra):
    r"""Runs ``sim2root.py`` on `raws` and returns the GRANDROOT directory.

    Parameters
    ----------
    raws : list of pathlib.Path
        The rawroot files, in order.
    out : pathlib.Path
        Parent directory for the output; created here.
    *extra : str
        More command-line arguments.

    Returns
    -------
    tuple
        ``(directory, log)``: the single output directory ``sim2root.py``
        made, and its captured output.
    """
    out.mkdir()
    completed = _run([SIM2ROOT] + list(raws) + ['-o', out, '-se', '1', '-sl', 'GP300',
                                               '-s', 'Xiaodushan'] + list(extra), out.parent)
    dirs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(dirs) == 1, 'expected one output directory, found %r' % dirs
    return dirs[0], completed.stdout + completed.stderr


def _tree(directory, pattern, name):
    r"""Returns the branches of tree `name` in the one file matching `pattern`.

    Parameters
    ----------
    directory : pathlib.Path
        Where to look.
    pattern : str
        Glob matching exactly one file.
    name : str
        The tree name.

    Returns
    -------
    callable-by-key
        ``result[branch]`` is that branch as a numpy array, one element per
        entry, read on access.
    """
    import uproot

    files = sorted(directory.glob(pattern))
    assert len(files) == 1, 'expected one %s in %s, found %r' % (pattern, directory, files)
    tree = uproot.open(str(files[0]))[name]

    class _Branches:
        def __getitem__(self, branch):
            return tree[branch].array(library='np')

    return _Branches()


@pytest.fixture(scope='module')
def work(tmp_path_factory):
    r"""Returns a directory holding the no-antenna rawroot file, ``noant.rawroot``.

    Built once per module: the fixture minus every ``a*.trace``, ``antpos.dat``
    and the rows of the ``.sry`` antenna table, converted.  ZHAireS lists no
    antenna when none was selected; a ``.sry`` that lists antennas without their
    trace files is a damaged simulation, which the converter refuses (#242).
    """
    base = tmp_path_factory.mktemp('no_antenna')
    folder = base / FIXTURE.name
    folder.mkdir()
    for path in FIXTURE.iterdir():
        if path.suffix == '.trace' or path.name == 'antpos.dat':
            continue
        if path.suffix == '.sry':
            lines = path.read_text().splitlines(keepends=True)
            start = next(i for i, line in enumerate(lines) if 'Antenna|      Label' in line) + 1
            end = start
            while len(lines[end].split()) == 6:
                end += 1
            (folder / path.name).write_text(''.join(lines[:start] + lines[end:]))
            continue
        shutil.copy(str(path), str(folder / path.name))
    assert not list(folder.glob('*.trace'))
    _convert(folder, base / 'noant.rawroot')
    return base


@pytest.fixture(scope='module')
def grandroot(work):
    r"""Returns ``(directory, log)`` of ``sim2root.py`` run on the no-antenna file."""
    return _sim2root([work / 'noant.rawroot'], work / 'grandroot')


def test_the_converter_keeps_a_shower_that_hit_no_antenna(work):
    r"""The rawroot file holds the shower, and an efield entry with no antenna."""
    shower = _tree(work, 'noant.rawroot', 'trawshower')
    efield = _tree(work, 'noant.rawroot', 'trawefield')
    meta = _tree(work, 'noant.rawroot', 'trawmeta')

    assert len(shower['event_number']) == 1
    assert int(shower['event_number'][0]) == EVENT_ID
    # The shower information survives: 3.8 EeV, zenith 51.6 deg, per the name
    assert float(np.ravel(shower['energy_primary'][0])[0]) == pytest.approx(3.8e9, rel=0.01)
    assert float(shower['zenith'][0]) == pytest.approx(51.64, abs=0.01)

    assert len(efield['event_number']) == 1 and len(meta['event_number']) == 1
    assert int(efield['du_count'][0]) == 0
    for branch in ('du_id', 'du_x', 'du_y', 'du_z', 't_0', 'trace_x', 'trace_y', 'trace_z'):
        assert len(efield[branch][0]) == 0, '%s is not empty' % branch


def test_sim2root_writes_the_event_with_du_count_zero(work, grandroot):
    r"""The event reaches every event file; the efield entry is empty; the run has no DU."""
    grandroot, log = grandroot
    shower = _tree(grandroot, 'shower_*.root', 'tshower')
    showersim = _tree(grandroot, 'showersim_*.root', 'tshowersim')
    efield = _tree(grandroot, 'efield_*.root', 'tefield')
    run = _tree(grandroot, 'run_*.root', 'trun')

    for tree in (shower, showersim, efield):
        assert list(tree['event_number']) == [1]
    raw = _tree(work, 'noant.rawroot', 'trawshower')
    assert float(np.ravel(shower['energy_primary'][0])[0]) == pytest.approx(float(np.ravel(raw['energy_primary'][0])[0]))
    assert float(shower['zenith'][0]) == pytest.approx(float(raw['zenith'][0]))

    assert int(efield['du_count'][0]) == 0
    for branch in ('du_id', 'du_seconds', 'du_nanoseconds', 'trace', 'trigger_position'):
        assert len(efield[branch][0]) == 0, '%s is not empty' % branch
    assert len(run['du_id'][0]) == 0
    assert len(run['du_xyz'][0]) == 0

    # And it says what it did
    assert 'hit no antenna' in log


def test_sim2root_mixes_a_miss_with_a_hit_in_either_order(work):
    r"""A no-antenna file next to a normal one: both events kept, the run gets the hit's DUs."""
    full = work / 'full.rawroot'
    if not full.is_file():
        _convert(FIXTURE, full)
    noant = work / 'noant.rawroot'

    for name, order in (('miss_first', [noant, full]), ('hit_first', [full, noant])):
        out, _ = _sim2root(order, work / name)
        efield = _tree(out, 'efield_*.root', 'tefield')
        run = _tree(out, 'run_*.root', 'trun')

        counts = [int(c) for c in efield['du_count']]
        expected = [0, 5] if order[0] is noant else [5, 0]
        assert counts == expected, '%s: du_count %r' % (name, counts)
        assert list(run['du_id'][0]) == [14, 21, 27, 28, 35]
        assert len(_tree(out, 'shower_*.root', 'tshower')['event_number']) == 2


def test_the_readers_load_the_event(grandroot):
    r"""``get_file_event`` loads it; the one method that needs traces says why it cannot."""
    from grand.dataio.root_files import get_file_event

    grandroot, _ = grandroot
    efield = sorted(grandroot.glob('efield_*.root'))[0]
    reader = get_file_event(str(efield))
    reader.load_event_idx(0)

    assert int(reader.get_du_count()) == 0
    assert reader.traces.shape == (0, 3, 0)
    assert reader.get_size_trace() == 0
    du_ns, t_ref = reader.get_du_nanosec_ordered()
    assert du_ns.shape == (0,) and np.isnan(t_ref)
    assert float(reader.get_simu_parameters()['zenith']) == pytest.approx(51.64, abs=0.01)

    with pytest.raises(ValueError, match='no antenna'):
        reader.get_obj_handling3dtraces()


class _NoModel:
    r"""Stands in for the antenna and RF-chain models.

    They need ``data/detector``, which an event with no antenna never reads.
    """

    def __init__(self, *args, **kwargs):
        pass


def test_efield2voltage_and_voltage2adc_write_the_event(grandroot, tmp_path, monkeypatch):
    r"""Voltage and ADC files each get the event, with du_count 0."""
    import grand.sim.efield2voltage as e2v

    grandroot, _ = grandroot
    for name in ('AntennaModel', 'RFChain', 'RFChainNut', 'RFChain_gaa'):
        monkeypatch.setattr(e2v, name, _NoModel)

    work = tmp_path / grandroot.name
    shutil.copytree(str(grandroot), str(work))
    signal = e2v.Efield2Voltage(str(work), 'voltage_1-1_L0_0000.root',
                                output_directory=str(work), seed=0)
    signal.compute_voltage()

    voltage = _tree(work, 'voltage_*.root', 'tvoltage')
    assert list(voltage['event_number']) == [1]
    assert int(voltage['du_count'][0]) == 0
    assert len(voltage['du_id'][0]) == 0 and len(voltage['trace'][0]) == 0

    completed = _run([VOLTAGE2ADC, work], work)
    assert 'no antenna' in completed.stdout + completed.stderr
    adc = _tree(work, 'adc_*.root', 'tadc')
    assert list(adc['event_number']) == [1]
    assert int(adc['du_count'][0]) == 0
    assert len(adc['trace_ch'][0]) == 0


CONVERT_EFIELD = ROOT / 'scripts' / 'convert_efield2efield.py'


def test_efield2efield_writes_the_event(grandroot, tmp_path):
    r"""#248: convert_efield2efield crashed on an event with no antenna."""
    grandroot, _ = grandroot
    work = tmp_path / grandroot.name
    shutil.copytree(str(grandroot), str(work))
    completed = _run([CONVERT_EFIELD, work], work)
    assert 'no antenna' in completed.stdout + completed.stderr
    efield = _tree(work, 'efield_*L1*.root', 'tefield')
    assert list(efield['event_number']) == [1]
    assert int(efield['du_count'][0]) == 0
