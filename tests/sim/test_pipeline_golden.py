# -*- coding: utf-8 -*-
r"""The whole chain, pinned to the answer it gives today.

Every other test in this suite checks GRANDlib against itself or against its
own inputs.  Even the galactic-noise check, which rebuilds the expected level
from the shipped tables, establishes that the code implements the documented
relation -- not that the composition of all the stages is right.  Nothing
would notice if the chain as a whole began producing systematically different
numbers.

This file is that notice.  It runs shower to voltage on a fixed input with a
fixed seed and compares against a stored result.

**It is not validation, and should not be described as one.**  It locks in
what the code does today; it cannot tell you that is correct.  What it gives
you is that any future change to the answer is *visible and deliberate*, and
that the answer on a particular date is recorded.

**Why it exists now.**  Phase 6 of the recovery plan rewrites this exact
chain -- pure kernel, configuration objects, ROOT pushed to the edges,
``Efield2Voltage`` reimplemented over the three -- and its stated goal is that
behaviour does not change.  A regression built before that refactor is what
makes the claim checkable.  Built afterwards it would only bless whatever the
refactor produced.

Regenerating, which should be rare and always deliberate::

    python tests/sim/test_pipeline_golden.py --write

Do that only when the answer is *meant* to change, and say why in the commit
message.  The stored file records the version, date and configuration that
produced it, so a later reader can tell what they are comparing against --
the lesson of three ``.npy`` tables that arrived with no provenance at all.
"""

import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tests.sim.test_pipeline_end_to_end import (  # noqa: E402
    N_DU, N_SAMPLES, PEAK_NS, SITE, T_BIN_NS, WIDTH_NS)

GOLDEN = pathlib.Path(__file__).with_name('pipeline_golden.npz')

#: Fixed across the reference and every comparison.  A different seed is a
#: different answer, so this is part of the contract, not a detail.
SEED = 0

#: The configuration the reference was produced under.  Stored in the file and
#: compared on load: a changed default is a changed answer, and should fail
#: here rather than silently shift the traces.
PARAMS = {'add_noise': True, 'add_rf_chain': True, 'lst': 18.0}


def _build_input(directory):
    r"""Writes the reference electric-field file and returns its path.

    Three detection units carrying a Gaussian pulse at a different amplitude
    on each arm -- deliberately asymmetric, so that a swapped component or a
    mis-ordered unit changes the answer instead of cancelling out.

    Parameters
    ----------
    directory : pathlib.Path
        Where to write it.

    Returns
    -------
    str
    """
    from grand.dataio.event_trees import TEfield, TShower
    from grand.dataio.run_trees import TRun

    path = str(pathlib.Path(directory) / 'efield.root')

    run = TRun(path)
    run.run_number = 0
    run.du_id = list(range(N_DU))
    run.du_xyz = [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [0.0, 500.0, 0.0]]
    run.t_bin_size = [T_BIN_NS] * N_DU
    run.origin_geoid = SITE
    run.fill()
    run.write()

    t_ns = np.arange(N_SAMPLES) * T_BIN_NS
    pulse = np.exp(-((t_ns - PEAK_NS) ** 2) / (2 * WIDTH_NS ** 2))
    trace = np.stack([np.stack([pulse * a for a in (1.0, 0.6, 0.2)])
                      for _ in range(N_DU)]).astype(np.float32)

    efield = TEfield(path)
    efield.run_number, efield.event_number = 0, 0
    efield.du_id = list(range(N_DU))
    efield.du_nanoseconds = [0] * N_DU
    efield.du_seconds = [0] * N_DU
    efield.trace = trace
    efield.fill()
    efield.write()

    shower = TShower(path)
    shower.run_number, shower.event_number = 0, 0
    shower.zenith, shower.azimuth = 85.0, 0.0
    shower.energy_primary = 3.98e9
    shower.shower_core_pos = [0.0, 0.0, 1200.0]
    shower.xmax_pos_shc = [0.0, 0.0, 10000.0]
    shower.fill()
    shower.write()
    return path


def _run_chain(directory):
    r"""Runs shower to voltage once and returns the traces.

    Parameters
    ----------
    directory : pathlib.Path
        Scratch space for the input and output files.

    Returns
    -------
    ndarray, shape (N_DU, 3, N_SAMPLES)
    """
    from grand import Efield2Voltage

    efield = _build_input(directory)
    signal = Efield2Voltage(efield, str(pathlib.Path(directory) / 'v.root'),
                            seed=SEED)
    signal.params.update(PARAMS)
    signal.compute_voltage()
    return np.asarray(signal.vout, dtype=np.float64)


def _provenance():
    r"""Returns what produced a reference, as a JSON string.

    Recorded inside the file rather than alongside it, so that a copy of the
    file carries its own explanation.

    Returns
    -------
    str
    """
    try:
        version = importlib.metadata.version('grand')
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown'
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=pathlib.Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        commit = 'unknown'
    return json.dumps({
        'grandlib_version': version,
        'commit': commit,
        'seed': SEED,
        'params': PARAMS,
        'note': 'Reference output of the shower-to-voltage chain. Locks in '
                'behaviour; does not validate it. Regenerate only when the '
                'answer is meant to change.',
    }, sort_keys=True)


@pytest.fixture(scope='module')
def traces(tmp_path_factory):
    r"""Runs the chain once for the whole module."""
    return _run_chain(tmp_path_factory.mktemp('golden'))


def test_the_reference_exists():
    r"""There is a stored answer to compare against.

    Skipped rather than failed when absent, so that a fresh checkout without
    the file reports the reason instead of a comparison error.
    """
    if not GOLDEN.exists():
        pytest.skip('no reference stored; create it with '
                    '`python %s --write`' % __file__)
    assert GOLDEN.stat().st_size > 0


def test_the_reference_says_what_produced_it():
    r"""The stored file carries its own provenance.

    Three ``.npy`` tables arrived in this repository with no record of how
    they were made, and closing that gap took a round trip to their author.
    A reference file that cannot say which code wrote it has the same defect,
    and is worse, because its whole purpose is to be compared against.
    """
    if not GOLDEN.exists():
        pytest.skip('no reference stored')
    with np.load(GOLDEN, allow_pickle=False) as data:
        assert 'provenance' in data, 'the reference carries no provenance'
        recorded = json.loads(str(data['provenance']))

    for key in ('grandlib_version', 'commit', 'seed', 'params'):
        assert key in recorded, 'provenance is missing %r' % key
    assert recorded['seed'] == SEED, (
        'the reference was made with seed %r, this test uses %r'
        % (recorded['seed'], SEED))
    assert recorded['params'] == PARAMS, (
        'the reference was made with %r, this test uses %r — a changed '
        'default is a changed answer, and has to be deliberate'
        % (recorded['params'], PARAMS))


def test_the_chain_reproduces_the_reference(traces):
    r"""Shower to voltage gives the stored answer, bin for bin.

    The tolerance is tight rather than exact.  The same seed on the same
    platform reproduces bit-for-bit, but NumPy and BLAS are entitled to
    reassociate floating-point sums between versions, and a test that failed
    on a library upgrade would be noise.  ``1e-9`` relative is far below any
    change that could matter physically and far above that reassociation.

    A failure here means the chain now answers differently.  That may be
    correct -- the galactic-noise fix on 2026-09-07 changed every voltage by
    :math:`\sqrt2`, legitimately -- but it must be deliberate.  Regenerate
    with ``--write`` and say why in the commit message.
    """
    if not GOLDEN.exists():
        pytest.skip('no reference stored')
    with np.load(GOLDEN, allow_pickle=False) as data:
        stored = data['voltage']
        recorded = json.loads(str(data['provenance']))

    assert traces.shape == stored.shape, (
        'the chain now returns %s, the reference holds %s'
        % (traces.shape, stored.shape))

    if not np.allclose(traces, stored, rtol=1e-9, atol=0.0):
        worst = np.unravel_index(np.argmax(np.abs(traces - stored)),
                                 stored.shape)
        pytest.fail(
            'the chain no longer reproduces the reference made by GRANDlib '
            '%s at %s. Largest disagreement at du=%d arm=%d sample=%d: '
            '%.6g against %.6g. If this change is intended, regenerate with '
            '`python %s --write` and say why in the commit message.'
            % (recorded.get('grandlib_version'), recorded.get('commit'),
               worst[0], worst[1], worst[2],
               traces[worst], stored[worst], __file__))


def test_the_reference_is_not_trivial(traces):
    r"""The stored traces carry signal, not zeros.

    A reference of all zeros would compare equal to another run of all zeros
    and look like a passing regression while testing nothing.
    """
    assert np.isfinite(traces).all(), 'the chain produced non-finite values'
    assert np.abs(traces).max() > 1.0, (
        'the traces peak at %.3g uV, which is too small to be a signal'
        % np.abs(traces).max())
    per_arm = np.abs(traces).max(axis=(0, 2))
    assert (per_arm > 0).all(), 'an arm is empty: %s' % per_arm


if __name__ == '__main__':
    if '--write' not in sys.argv:
        print(__doc__)
        raise SystemExit('pass --write to regenerate the reference')
    import tempfile
    out = _run_chain(pathlib.Path(tempfile.mkdtemp()))
    np.savez_compressed(GOLDEN, voltage=out, provenance=_provenance())
    print('wrote %s  %s  %.1f kB'
          % (GOLDEN, out.shape, GOLDEN.stat().st_size / 1024))
    print(_provenance())
