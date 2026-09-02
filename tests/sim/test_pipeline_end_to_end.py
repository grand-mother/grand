# -*- coding: utf-8 -*-
r"""Exercises the whole chain, on a fixture built here rather than shipped.

``tests/sim/test_efield2voltage.py`` reads ``data/test_efield.root``, which is
not in version control -- ``data/.gitignore`` excludes everything -- and is not
fetched by ``env/setup.sh``.  So the one test that covered the pipeline could
not run on a fresh checkout or in CI, and asserted only that an output file
appeared, which is all you can assert with no reliable input.

This builds its input instead, from the tree classes themselves.  That costs
nothing in repository size, cannot drift from the schema, and makes the
contents of the fixture visible in the test rather than opaque.
"""

import os

import numpy as np
import pytest

from grand import Efield2Voltage
from grand.dataio.event_trees import TEfield, TShower
from grand.dataio.run_trees import TRun

N_DU = 3
N_SAMPLES = 512
T_BIN_NS = 0.5
PEAK_NS = 120.0
WIDTH_NS = 4.0
# The GRANDProto300 site, as used throughout the documentation.
SITE = [40.98, 93.95, 1200.0]


@pytest.fixture(scope='module')
def efield_file(tmp_path_factory):
    r"""Returns the path of a small electric-field file, built for this test.

    Three detection units, each carrying a Gaussian pulse of a different
    amplitude on the three arms, so that a component swap or a unit ordering
    error shows up as a changed answer rather than a symmetric one.

    Returns
    -------
    str
        Path to the file.
    """
    path = str(tmp_path_factory.mktemp('pipeline') / 'efield.root')

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


def _run(efield_file, tmp_path, **params):
    r"""Runs the chain on `efield_file` and returns the simulator.

    Parameters
    ----------
    efield_file : str
        Input file.
    tmp_path : pathlib.Path
        Directory for the output.
    **params
        Overrides applied to ``signal.params`` before computing.

    Returns
    -------
    Efield2Voltage
        The simulator, after ``compute_voltage()``.
    """
    os.makedirs(str(tmp_path), exist_ok=True)
    out = str(tmp_path / 'voltage.root')
    signal = Efield2Voltage(efield_file, out, seed=0)
    signal.params.update(params)
    signal.compute_voltage()
    signal.output_path = out
    return signal


def test_default_params_are_complete(efield_file, tmp_path):
    r"""A default-constructed simulator can run without setting any parameter.

    Four keys -- ``resample_to_mhz``, ``extend_to_us``,
    ``calibration_smearing_sigma`` and ``add_jitter_ns`` -- used to be set
    only by ``scripts/convert_efield2voltage.py``, so the command line worked
    while the documented Python usage raised ``KeyError`` on the first call
    to :meth:`compute_voltage`.  That is the usage shown in Listing 4 of the
    GRANDlib paper.
    """
    out = str(tmp_path / 'voltage.root')
    signal = Efield2Voltage(efield_file, out, seed=0)
    signal.compute_voltage()          # must not raise KeyError
    assert os.path.exists(out)


def test_chain_produces_finite_voltages(efield_file, tmp_path):
    r"""The chain runs and gives finite voltages of the expected shape."""
    signal = _run(efield_file, tmp_path)
    vout = np.asarray(signal.vout)
    assert vout.shape == (N_DU, 3, N_SAMPLES), 'unexpected output shape'
    assert np.isfinite(vout).all(), 'the chain produced non-finite voltages'
    assert np.abs(vout).max() > 0.0, 'the chain produced an all-zero output'


def test_seed_makes_the_run_reproducible(efield_file, tmp_path):
    r"""The same seed gives the same voltages; a different one does not.

    The noise is the only stochastic part, so this also confirms that it is
    actually being added.
    """
    a = np.asarray(_run(efield_file, tmp_path / 'a', add_noise=True).vout)
    b = np.asarray(_run(efield_file, tmp_path / 'b', add_noise=True).vout)
    assert np.array_equal(a, b), 'the same seed gave different output'


def test_noise_and_chain_each_change_the_answer(efield_file, tmp_path):
    r"""Disabling noise, or the RF chain, changes the result.

    A guard against a stage silently not running: if either flag made no
    difference, the flag would be decorative.
    """
    full = np.asarray(_run(efield_file, tmp_path / 'f').vout)
    quiet = np.asarray(_run(efield_file, tmp_path / 'q', add_noise=False).vout)
    bare = np.asarray(_run(efield_file, tmp_path / 'b',
                           add_noise=False, add_rf_chain=False).vout)

    assert not np.allclose(full, quiet), 'add_noise made no difference'
    assert not np.allclose(quiet, bare), 'add_rf_chain made no difference'


def test_rf_chain_amplifies(efield_file, tmp_path):
    r"""The RF chain increases the signal amplitude.

    With a 20 dB variable-gain amplifier the chain is a net amplifier across
    the band, so the output must be larger than the open-circuit voltage.
    Figure 8 of the GRANDlib paper shows the transfer function this reflects.
    """
    with_chain = np.asarray(_run(efield_file, tmp_path / 'w',
                                 add_noise=False).vout)
    without = np.asarray(_run(efield_file, tmp_path / 'o',
                              add_noise=False, add_rf_chain=False).vout)
    assert np.abs(with_chain).max() > np.abs(without).max(), (
        'the RF chain did not amplify: %.3g vs %.3g'
        % (np.abs(with_chain).max(), np.abs(without).max()))


def test_output_file_is_readable(efield_file, tmp_path):
    r"""The voltages written back can be read as a voltage tree."""
    from grand.dataio.event_trees import TVoltage

    signal = _run(efield_file, tmp_path)
    voltage = TVoltage(signal.output_path)
    voltage.get_entry(0)
    trace = np.asarray(voltage.trace)
    assert trace.shape[0] == N_DU, 'wrong number of units in the output'
    assert np.isfinite(trace).all(), 'output file holds non-finite values'
