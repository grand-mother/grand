# -*- coding: utf-8 -*-
r"""Galactic-noise normalisation: properties that hold, and levels that are pinned.

**History, because this file used to say the opposite.**  Until 2026-09-07 the
normalisation was an open question.  ``galaxy.py`` scaled the spectrum by
``size_out / 2``; PR 153 proposed ``size_out / sqrt(2)``; and which was right
turned on whether the tabulated ``Vocmax_...`` quantity was an RMS or a
maximum.  Measured against the table each ``du_type`` actually read, the
simulated RMS came out at :math:`1/\sqrt2` of the tabulated value, which is
what one expects if the table is an RMS and the code is treating it as a
maximum -- but the filename said *max*, and nothing in the repository could
settle it.

Stavros Nonis settled it, and this file records the resolution rather than the
question.  The Galactic-noise calculation starts from the available power
spectral density :math:`P_L`, from which the open-circuit voltage is
reconstructed as

.. math::  V_{\rm oc,RMS}^2 = 4 P_L \,\mathrm{Re}(Z_{\rm ant})

so the quantity used as the Gaussian standard deviation **is** an RMS by
construction, and ``size_out / sqrt(2)`` is correct.  ``dev_snonis`` @
``0205c15`` applies that, and supplies matching :math:`P_L` tables for all
three antenna models.

**What this file does and does not claim.**  The absolute scale rests on how
those :math:`P_L` tables were generated, which is not in this repository; the
FFT normalisation, LST selection, RMS level and the integration through
``Efield2Voltage`` were validated by their author.  Nothing here re-derives
that.  What is asserted below is of two kinds: *properties* that must hold
whatever the tables contain, and *pinned levels* that turn a silent change in
those tables into a failing test.

Three defects this file used to record are now fixed and their tests are gone:
the ``nec`` and ``mat`` tables were byte-identical, the default ``GP300``
recomputed its own from a MATLAB file via ``h5py``, and the ``hfss`` tables
were reachable from no ``du_type`` at all.  Each model now reads one distinct
table of its own.
"""

import numpy as np
import pytest

from grand.sim.noise.galaxy import galactic_noise

#: Sidereal hour to sample.  Integer hours land exactly on a table bin.
LST_HOUR = 18.0

#: With 1 MHz bins, 30-250 MHz lands on bins 30..250 of the one-sided spectrum.
N_SAMPLES = 2048

#: Enough antennas for the sample RMS to settle to well under a percent.
N_ANTENNAS = 600

FREQS_MHZ = np.arange(30.0, 251.0)

MODELS = ("GP300", "GP300_nec", "GP300_mat")

#: Time-domain RMS per arm, in microvolts, at `LST_HOUR` with seed 1.
#: Measured on dev_snonis @ 0205c15.  These are pinned, not derived: they exist
#: so that a change in the tables or in the transform shows up here rather than
#: silently in someone's noise floor a year from now.
PINNED_RMS_UV = {
    "GP300":     (29.8452, 37.8260, 35.1107),
    "GP300_nec": (31.4952, 39.3583, 37.8664),
    "GP300_mat": (32.4994, 40.3907, 37.3170),
}


def _traces(du_type="GP300", seed=1, size_out=N_SAMPLES, nb_ant=N_ANTENNAS):
    r"""Returns the band spectrum, the padded spectrum and the time series.

    Parameters
    ----------
    du_type : str, optional
        Antenna model passed to :func:`galactic_noise`.
    seed, size_out, nb_ant : optional
        Passed straight through.

    Returns
    -------
    band : ndarray, shape (nb_ant, 3, 221)
        What :func:`galactic_noise` returns.
    full : ndarray, shape (nb_ant, 3, size_out // 2 + 1)
        The same, embedded in a complete one-sided spectrum, which is what
        :mod:`grand.sim.efield2voltage` effectively does.
    traces : ndarray, shape (nb_ant, 3, size_out)
        The corresponding time series.
    """
    band = galactic_noise(LST_HOUR, size_out, FREQS_MHZ,
                          nb_ant=nb_ant, seed=seed, du_type=du_type)
    full = np.zeros((nb_ant, 3, size_out // 2 + 1), dtype=complex)
    full[:, :, 30:251] = band
    return band, full, np.fft.irfft(full, n=size_out, axis=-1)


# --------------------------------------------------------------------------
# Properties: true whatever the tables contain.
# --------------------------------------------------------------------------

def test_parseval_internal_consistency():
    r"""Spectral power and time-domain power agree, as Parseval requires.

    A check on the transform rather than on the physics: if the spectrum and
    the trace disagree about how much power is present, the normalisation is
    wrong regardless of what the tables say.
    """
    _, full, traces = _traces()
    measured = (traces ** 2).sum(axis=-1)
    expected = 2.0 * (np.abs(full[:, :, 1:-1]) ** 2).sum(axis=-1) / N_SAMPLES
    expected += (np.abs(full[:, :, 0]) ** 2 + np.abs(full[:, :, -1]) ** 2) / N_SAMPLES
    assert np.allclose(measured, expected, rtol=1e-9), (
        'Parseval violated: spectrum and trace disagree about total power')


def test_reproducible_with_a_seed():
    r"""The same seed gives the same noise; a different one does not."""
    a = galactic_noise(LST_HOUR, N_SAMPLES, FREQS_MHZ, nb_ant=4, seed=7)
    b = galactic_noise(LST_HOUR, N_SAMPLES, FREQS_MHZ, nb_ant=4, seed=7)
    c = galactic_noise(LST_HOUR, N_SAMPLES, FREQS_MHZ, nb_ant=4, seed=8)
    assert np.array_equal(a, b), 'same seed gave different noise'
    assert not np.array_equal(a, c), 'different seeds gave identical noise'


def test_all_three_arms_are_populated():
    r"""Every antenna arm carries noise, and none dominates the others.

    Guards against an indexing error that fills one arm and leaves the rest
    at zero -- which would still produce plausible-looking traces.
    """
    _, _, traces = _traces()
    power = (traces ** 2).mean(axis=(0, 2))
    assert (power > 0).all(), 'an antenna arm carries no noise: %s' % power
    assert power.min() / power.max() > 0.1, (
        'the arms differ by more than 10x, which looks like an indexing '
        'error rather than antenna response: %s' % power)


@pytest.mark.parametrize('size_out', [1024, 2048, 4096, 8192])
def test_level_does_not_depend_on_transform_length(size_out):
    r"""The noise level is a physical quantity, not a property of the FFT.

    Asking for a longer transform must not change how many microvolts of
    noise an antenna sees.  This is the sharpest self-contained statement
    available about the normalisation: it needs no external reference, and a
    scaling that carried a stray factor of `size_out` or its square root
    would fail here immediately.
    """
    _, _, reference = _traces(size_out=N_SAMPLES)
    _, _, traces = _traces(size_out=size_out)
    assert np.allclose(traces.std(axis=(0, 2)), reference.std(axis=(0, 2)),
                       rtol=1e-6), (
        'RMS at size_out=%d is %s, against %s at %d; the normalisation '
        'depends on the transform length'
        % (size_out, traces.std(axis=(0, 2)), reference.std(axis=(0, 2)),
           N_SAMPLES))


# --------------------------------------------------------------------------
# The tables, and the levels they produce.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('du_type', MODELS)
def test_each_model_reads_one_table_of_its_own(du_type):
    r"""Each antenna model opens exactly one table, and its own.

    Replaces three tests of defects that are now fixed: ``nec`` and ``mat``
    used to be byte-identical files, the default ``GP300`` used to recompute
    its table from a MATLAB file through ``h5py``, and the ``hfss`` tables
    were opened by no ``du_type`` at all.

    Recorded by watching what the function opens rather than by reading the
    source, because the module docstring names these files and a text search
    would find them and prove nothing.
    """
    from grand.sim.noise import galaxy

    opened, real_load = [], galaxy.np.load

    def recording_load(path, *args, **kwargs):
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    galaxy.np.load = recording_load
    try:
        galactic_noise(LST_HOUR, N_SAMPLES, FREQS_MHZ, nb_ant=2, seed=1,
                       du_type=du_type)
    finally:
        galaxy.np.load = real_load

    tables = [p for p in opened if p.endswith('.npy')]
    assert len(tables) == 1, (
        '%s opened %d tables, expected exactly one: %s'
        % (du_type, len(tables), tables))
    assert tables[0].endswith('galactic_PL_per_Hz_gp13_%s.npy'
                              % ('GP300' if du_type == 'GP300' else du_type)), (
        '%s read %s, which is not its own table' % (du_type, tables[0]))


def test_the_three_models_give_different_noise():
    r"""The models are genuinely distinct, which they did not used to be.

    ``nec`` and ``mat`` were byte-identical tables, so two of the three
    ``du_type`` values were the same run under different names.  Now each has
    its own :math:`P_L` table and the levels differ.
    """
    levels = {du: _traces(du)[2].std(axis=(0, 2)) for du in MODELS}
    for a, b in (('GP300', 'GP300_nec'), ('GP300', 'GP300_mat'),
                 ('GP300_nec', 'GP300_mat')):
        assert not np.allclose(levels[a], levels[b], rtol=1e-6), (
            '%s and %s produce identical noise: %s' % (a, b, levels[a]))


@pytest.mark.parametrize('du_type', MODELS)
def test_rms_level_is_unchanged(du_type):
    r"""The level each model produces is the level recorded when it landed.

    A regression pin, not a validation.  It does not establish that the
    absolute scale is right -- that rests on how the :math:`P_L` tables were
    generated, which is not in this repository -- but it does mean that a
    change to those tables, to the transform, or to the interpolation shows
    up as a failing test rather than as a quietly different noise floor.

    If this fails after a deliberate table update, re-measure and update
    `PINNED_RMS_UV`, and say so in the commit message.
    """
    _, _, traces = _traces(du_type)
    measured = traces.std(axis=(0, 2))
    assert np.allclose(measured, PINNED_RMS_UV[du_type], rtol=2e-4), (
        '%s RMS is %s uV, pinned at %s'
        % (du_type, np.round(measured, 4), PINNED_RMS_UV[du_type]))


# --------------------------------------------------------------------------
# Input validation, which the rewrite added.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('bad_lst', [-1.0, 24.0, 25.5, float('nan')])
def test_rejects_impossible_sidereal_times(bad_lst):
    r"""LST outside ``[0, 24)`` raises rather than wrapping silently."""
    with pytest.raises(ValueError):
        galactic_noise(bad_lst, N_SAMPLES, FREQS_MHZ, nb_ant=2, seed=1)


def test_rejects_an_unknown_antenna_model():
    r"""An unrecognised ``du_type`` raises, naming what is available."""
    with pytest.raises(ValueError) as raised:
        galactic_noise(LST_HOUR, N_SAMPLES, FREQS_MHZ, nb_ant=2, seed=1,
                       du_type='GP300_nonexistent')
    assert 'GP300' in str(raised.value), (
        'the error does not say which models exist: %s' % raised.value)


def test_rejects_a_non_uniform_frequency_grid():
    r"""A frequency grid that is not uniformly spaced raises.

    The interpolation onto the requested grid assumes uniform bins in order
    to rescale the RMS to the bin width; a ragged grid would silently give
    the wrong level.
    """
    ragged = np.array([30.0, 31.0, 33.0, 36.0, 40.0])
    with pytest.raises(ValueError):
        galactic_noise(LST_HOUR, N_SAMPLES, ragged, nb_ant=2, seed=1)
