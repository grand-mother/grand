# -*- coding: utf-8 -*-
r"""Checks the normalisation of the simulated Galactic noise.

There is a live disagreement about one constant.  ``grand/sim/noise/galaxy.py``
builds the spectrum as ``|amp| * size_out / 2``; the branch ``dev_snonis``
(PR 153) changes that to ``size_out / sqrt(2)``, which scales every simulated
noise voltage by about 1.41 and therefore moves every trigger threshold and
sensitivity estimate downstream of it.  Nothing in the suite could tell the two
apart, so this file measures rather than argues.

Two things are checked, and the distinction matters.

**Internal consistency** is a property of any correct implementation, whatever
convention it adopts: the time series obtained from the returned spectrum must
carry that spectrum's energy.  This is asserted.

**Agreement with the tabulated model** is the contested part, and it is
*recorded, not asserted*.  Measured on 2026-08-30 against
``data/noise/Vocmax_30-250MHz_uVperMHz_hfss.npy`` at LST 18 h, the simulated
RMS comes out at about **0.33** of the value Parseval gives for the table
with the current ``size_out/2``, and would be about **0.47** with
``size_out/sqrt(2)``.  Neither is 1.

So the √2 change moves the result towards the model but does not by itself
reconcile the two, and the remaining factor of roughly 2 is unexplained.  The
question that settles it is definitional and belongs to the authors of the
table: **is** ``Vocmax_..._uVperMHz`` **an RMS voltage spectral density, or a
maximum?**  If it is a maximum, the reference used here is wrong by a known
factor and the comparison shifts accordingly.

Until that is answered, ``test_matches_tabulated_model`` stays skipped with
the measurement in its message, so the number is visible without pretending
the target is agreed.
"""

import numpy as np
import pytest

from grand import grand_add_path_data
from grand.sim.noise.galaxy import galactic_noise, interpol_at_new_x

LST_HOUR = 18
N_SAMPLES = 2048          # with 1 MHz bins, 30-250 MHz lands on bins 30..250
N_ANTENNAS = 600          # enough for the sample RMS to settle to ~1%
FREQS_MHZ = np.arange(30.0, 251.0)


def _spectrum_and_traces(seed=1):
    r"""Returns the returned band, the full spectrum and its time series.

    Returns
    -------
    band : ndarray, shape (n_ant, 3, 221)
        What :func:`galactic_noise` returns.
    full : ndarray, shape (n_ant, 3, N/2+1)
        The same, embedded in a complete one-sided spectrum, which is what
        the caller in :mod:`grand.sim.efield2voltage` effectively does.
    traces : ndarray, shape (n_ant, 3, N)
        The corresponding time series.
    """
    band = galactic_noise(float(LST_HOUR), N_SAMPLES, FREQS_MHZ,
                          nb_ant=N_ANTENNAS, seed=seed)
    full = np.zeros((N_ANTENNAS, 3, N_SAMPLES // 2 + 1), dtype=complex)
    full[:, :, 30:251] = band
    return band, full, np.fft.irfft(full, n=N_SAMPLES, axis=-1)


def _model_rms():
    r"""Returns the per-arm RMS the tabulated model implies, in microvolts."""
    table = np.transpose(np.load(grand_add_path_data(
        "noise/Vocmax_30-250MHz_uVperMHz_hfss.npy")), (0, 2, 1))
    per_bin = table[:, :, LST_HOUR - 1] * np.sqrt(FREQS_MHZ[1] - FREQS_MHZ[0])
    amp = np.stack([interpol_at_new_x(np.arange(30.0, 251.0), per_bin[:, i],
                                      FREQS_MHZ) for i in range(3)], axis=1)
    return np.sqrt(np.sum(amp ** 2, axis=0))


def test_parseval_internal_consistency():
    r"""The time series carries the energy of the spectrum it came from.

    True for any normalisation convention, so this holds equally before and
    after the ``dev_snonis`` change and will hold for a rewrite such as
    ``refact_galaxy``.  It is the invariant that survives the decision.
    """
    _, full, traces = _spectrum_and_traces()
    measured = traces.var(axis=(0, 2))
    expected = (2.0 * np.abs(full) ** 2).sum(axis=-1).mean(axis=0) / N_SAMPLES ** 2
    assert np.allclose(measured, expected, rtol=1e-9), (
        'time-domain variance %s does not match the spectrum energy %s'
        % (measured, expected))


def test_reproducible_with_a_seed():
    r"""A fixed seed gives identical output, and different seeds do not."""
    a, _, _ = _spectrum_and_traces(seed=7)
    b, _, _ = _spectrum_and_traces(seed=7)
    c, _, _ = _spectrum_and_traces(seed=8)
    assert np.array_equal(a, b), 'same seed gave different noise'
    assert not np.array_equal(a, c), 'different seeds gave identical noise'


def test_all_three_arms_are_populated():
    r"""No arm is silently zero.

    The Z arm is vertical and sees less sky than X and Y, so it is the one a
    shape or transpose error would most plausibly blank without the result
    looking obviously wrong.
    """
    band, _, _ = _spectrum_and_traces()
    power = (np.abs(band) ** 2).mean(axis=(0, 2))
    assert (power > 0).all(), 'an antenna arm carries no noise: %s' % power
    assert power.min() / power.max() > 0.1, (
        'one arm is implausibly quieter than the others: %s' % power)


@pytest.mark.skip(reason='target under discussion; see the module docstring')
def test_matches_tabulated_model():
    r"""The simulated RMS reproduces the tabulated model, within tolerance.

    Skipped until it is settled whether the table is an RMS or a maximum.
    Run it directly to see the current numbers.
    """
    _, _, traces = _spectrum_and_traces()
    ratio = traces.std(axis=(0, 2)) / _model_rms()
    assert np.allclose(ratio, 1.0, rtol=0.05), (
        'simulated / tabulated = %s; expected 1.0' % np.round(ratio, 4))


if __name__ == '__main__':
    _, _, traces = _spectrum_and_traces()
    ratio = traces.std(axis=(0, 2)) / _model_rms()
    print('simulated / tabulated, per arm : %s' % np.round(ratio, 4))
    print('the same with size_out/sqrt(2) : %s' % np.round(ratio * np.sqrt(2), 4))
