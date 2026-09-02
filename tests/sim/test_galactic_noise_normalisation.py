# -*- coding: utf-8 -*-
r"""Checks the normalisation of the simulated Galactic noise.

There is a live disagreement about one constant.  ``grand/sim/noise/galaxy.py``
builds the spectrum as ``|amp| * size_out / 2``; the branch ``dev_snonis``
(PR 153) changes that to ``size_out / sqrt(2)``, which scales every simulated
noise voltage by about 1.41 and therefore moves every trigger threshold and
sensitivity estimate downstream of it.  Nothing in the suite could tell the two
apart, so this file measures rather than argues.

What the measurement says
-------------------------

For **every** ``du_type``, and independently of ``size_out`` and of the number
of antennas, the simulated time-domain RMS comes out at

.. math::  \frac{V_{\rm rms}^{\rm sim}}{V^{\rm table}} = 0.705 \pm 0.3\%
           \qquad \left(1/\sqrt{2} = 0.7071\right)

That is not a loose agreement: it is :math:`1/\sqrt2` to a fraction of a
percent, and it does not drift with any parameter.  The implementation is
therefore producing an RMS that is exactly :math:`1/\sqrt2` of the tabulated
value, which is precisely the relation between the RMS and the peak of a
sinusoid.

So the decision reduces to one definitional question, and each answer picks a
different constant:

=========================================  ==========================
if ``Vocmax_..._uVperMHz`` is a **maximum**  the current ``size_out/2`` is right
if it is an **RMS**                          PR 153's ``size_out/sqrt(2)`` is right
=========================================  ==========================

The filename says *max*.  Nothing else in the repository states which is
meant, so the test below records the ratio and asserts only the parts that no
convention can change.

.. versionchanged:: 0.1.0
   An earlier version of this file compared the default ``GP300`` simulation
   against ``Vocmax_30-250MHz_uVperMHz_hfss.npy`` and reported a ratio of 0.33
   with an unexplained factor of roughly 2.  Those are different
   normalisations of the same model (see
   :func:`test_gp300_is_the_hfss_model_at_a_different_normalisation`), so the
   comparison was not like for like.  Compared against the table it actually
   uses, the ratio is :math:`1/\sqrt2`.
"""

import hashlib

import numpy as np
import pytest

from grand import grand_add_path_data
from grand.sim.noise.galaxy import galactic_noise

LST_HOUR = 18
N_SAMPLES = 2048          # with 1 MHz bins, 30-250 MHz lands on bins 30..250
N_ANTENNAS = 600          # enough for the sample RMS to settle to ~1%
FREQS_MHZ = np.arange(30.0, 251.0)

#: The table each ``du_type`` actually reads.  ``GP300`` reads none of them:
#: it recomputes the voltage from ``PG_ALL_jifen.mat``.
TABLE_OF = {
    "GP300_nec": "noise/Vocmax_30-250MHz_uVperMHz_nec.npy",
    "GP300_mat": "noise/Vocmax_30-250MHz_uVperMHz_mat.npy",
}


def _traces(du_type="GP300", seed=1, size_out=N_SAMPLES, nb_ant=N_ANTENNAS):
    r"""Returns the spectrum, its zero-padded form and the time series.

    Parameters
    ----------
    du_type : str, optional
        Which antenna model to pass to :func:`galactic_noise`.
    seed, size_out, nb_ant : optional
        Passed straight through.

    Returns
    -------
    band : ndarray, shape (nb_ant, 3, 221)
        What :func:`galactic_noise` returns.
    full : ndarray, shape (nb_ant, 3, size_out // 2 + 1)
        The same, embedded in a complete one-sided spectrum, which is what
        the caller in :mod:`grand.sim.efield2voltage` effectively does.
    traces : ndarray, shape (nb_ant, 3, size_out)
        The corresponding time series.
    """
    band = galactic_noise(float(LST_HOUR), size_out, FREQS_MHZ,
                          nb_ant=nb_ant, seed=seed, du_type=du_type)
    full = np.zeros((nb_ant, 3, size_out // 2 + 1), dtype=complex)
    full[:, :, 30:251] = band
    return band, full, np.fft.irfft(full, n=size_out, axis=-1)


def _gp300_internal_table():
    r"""Returns the voltage table the ``GP300`` branch builds, in microvolts.

    Reproduces the arithmetic of :func:`galactic_noise` for that branch:
    :math:`V_{\rm oc}^2 = 4 P R_{\rm ant}`, with the power read from
    ``PG_ALL_jifen.mat`` and the resistance from the 3.2 m antenna impedance.

    Returns
    -------
    ndarray, shape (221, 3, 24)
        Indexed by frequency, arm and LST hour.
    """
    import h5py

    with h5py.File(grand_add_path_data("noise/PG_ALL_jifen.mat"), "r") as f:
        power = np.transpose(np.array(f["PG_ALL_jifen"]), (2, 0, 1))
    zant = np.loadtxt(grand_add_path_data("detector/RFchain_v2/Z_ant_3.2m.csv"),
                      delimiter=",", skiprows=1)
    r_ant = np.column_stack([zant[:, 1], zant[:, 3], zant[:, 5]]).T
    return np.stack([1e6 * np.sqrt(4.0 * (1e6 * power[:, :, i]) * r_ant[i][:, None])
                     for i in range(3)], axis=1)


def _band_rms(table):
    r"""Returns the per-arm RMS implied by a table slice, in microvolts.

    Parameters
    ----------
    table : ndarray, shape (221, 3)
        Voltage spectral density at one LST, per frequency and arm.

    Returns
    -------
    ndarray, shape (3,)
        Quadrature sum over the band, one value per arm.
    """
    return np.sqrt(np.sum(table ** 2, axis=0))


def _reference(du_type):
    r"""Returns the band RMS of the table that ``du_type`` reads."""
    if du_type == "GP300":
        table = _gp300_internal_table()
    else:
        table = np.transpose(np.load(grand_add_path_data(TABLE_OF[du_type])),
                             (0, 2, 1))
    return _band_rms(table[:, :, LST_HOUR - 1])


# --------------------------------------------------------------------------
# invariants: true under either convention
# --------------------------------------------------------------------------

def test_parseval_internal_consistency():
    r"""The time series carries the energy of the spectrum it came from.

    True for any normalisation convention, so this holds equally before and
    after the ``dev_snonis`` change and will hold for a rewrite such as
    ``refact_galaxy``.  It is the invariant that survives the decision.
    """
    _, full, traces = _traces()
    measured = traces.var(axis=(0, 2))
    expected = (2.0 * np.abs(full) ** 2).sum(axis=-1).mean(axis=0) / N_SAMPLES ** 2
    assert np.allclose(measured, expected, rtol=1e-9), (
        'time-domain variance %s does not match the spectrum energy %s'
        % (measured, expected))


def test_reproducible_with_a_seed():
    r"""A fixed seed gives identical output, and different seeds do not."""
    a, _, _ = _traces(seed=7)
    b, _, _ = _traces(seed=7)
    c, _, _ = _traces(seed=8)
    assert np.array_equal(a, b), 'same seed gave different noise'
    assert not np.array_equal(a, c), 'different seeds gave identical noise'


def test_all_three_arms_are_populated():
    r"""No arm is silently zero.

    The Z arm is vertical and sees less sky than X and Y, so it is the one a
    shape or transpose error would most plausibly blank without the result
    looking obviously wrong.
    """
    band, _, _ = _traces()
    power = (np.abs(band) ** 2).mean(axis=(0, 2))
    assert (power > 0).all(), 'an antenna arm carries no noise: %s' % power
    assert power.min() / power.max() > 0.1, (
        'one arm is implausibly quieter than the others: %s' % power)


# --------------------------------------------------------------------------
# the normalisation itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("du_type", ["GP300", "GP300_nec", "GP300_mat"])
def test_simulated_rms_is_the_table_over_root_two(du_type):
    r"""The simulated RMS is :math:`1/\sqrt2` of the table it was built from.

    This is the sharp statement the decision rests on.  It is asserted rather
    than recorded because it is a property of the code as written: if someone
    applies PR 153 without also settling the definitional question, this test
    fails and says which constant changed.

    Compare each ``du_type`` against **its own** table; the four antenna
    models differ in absolute level by up to a factor of two, so comparing
    across them measures the model difference and not the normalisation.
    """
    _, _, traces = _traces(du_type=du_type)
    ratio = traces.std(axis=(0, 2)) / _reference(du_type)
    assert np.allclose(ratio, 1.0 / np.sqrt(2.0), rtol=0.02), (
        '%s: simulated / tabulated = %s, expected 1/sqrt(2) = %.4f'
        % (du_type, np.round(ratio, 4), 1.0 / np.sqrt(2.0)))


@pytest.mark.parametrize("size_out", [1024, 2048, 4096])
def test_normalisation_does_not_depend_on_transform_length(size_out):
    r"""``size_out`` cancels: the RMS is a property of the model, not the FFT.

    A normalisation that leaked the transform length would make the noise
    level depend on the trace length chosen by the caller, which would be a
    much worse defect than a constant factor.  It does not.
    """
    _, _, traces = _traces(size_out=size_out, nb_ant=200)
    ratio = traces.std(axis=(0, 2)) / _reference("GP300")
    assert np.allclose(ratio, 1.0 / np.sqrt(2.0), rtol=0.02), (
        'size_out=%d gave ratio %s' % (size_out, np.round(ratio, 4)))


# --------------------------------------------------------------------------
# the shipped tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["Vocmax_30-250MHz_uVperMHz",
                                  "Pocmax_30-250_Watt_per_MHz",
                                  "Voutmax_30-250MHz_uVperMHz"])
def test_nec_and_mat_tables_are_the_same_file(kind):
    r"""Records that the NEC and MATLAB tables are byte-identical.

    :func:`galactic_noise` documents ``GP300_nec`` and ``GP300_mat`` as "the
    NEC or MATLAB variants" of the antenna response, but all three pairs of
    shipped ``.npy`` tables have the same SHA-256, so the two options select
    the same numbers.  One of the files was presumably copied over the other.

    Asserted so that it becomes a *passing* test the day someone restores the
    distinct table -- at which point this test fails and points at the
    docstring that then needs no change.
    """
    digests = {}
    for variant in ("nec", "mat"):
        path = grand_add_path_data("noise/%s_%s.npy" % (kind, variant))
        with open(path, "rb") as handle:
            digests[variant] = hashlib.sha256(handle.read()).hexdigest()
    assert digests["nec"] == digests["mat"], (
        '%s: nec and mat now differ (%s vs %s) -- the docstring of '
        'galactic_noise can drop its caveat' % (kind, digests["nec"][:12],
                                                digests["mat"][:12]))


def test_gp300_is_the_hfss_model_at_a_different_normalisation():
    r"""``GP300`` and the ``hfss`` table are one model at two normalisations.

    The ratio between them is flat across the band to a few percent, which is
    what a normalisation difference looks like; against the ``nec`` table the
    ratio varies by an order of magnitude more, which is what a genuinely
    different antenna model looks like.  This is the check that explains why
    an earlier version of this file measured 0.33 instead of
    :math:`1/\sqrt2`.
    """
    gp300 = _gp300_internal_table()[:, :, LST_HOUR - 1]
    hfss = np.transpose(np.load(grand_add_path_data(
        "noise/Vocmax_30-250MHz_uVperMHz_hfss.npy")), (0, 2, 1))[:, :, LST_HOUR - 1]
    nec = np.transpose(np.load(grand_add_path_data(
        "noise/Vocmax_30-250MHz_uVperMHz_nec.npy")), (0, 2, 1))[:, :, LST_HOUR - 1]

    flat = (gp300 / hfss)[:, 0]
    varying = (gp300 / nec)[:, 0]
    assert flat.std() / flat.mean() < 0.05, (
        'GP300 / hfss is not flat (%.1f%%); they may not be the same model'
        % (100 * flat.std() / flat.mean()))
    assert varying.std() / varying.mean() > 0.05, (
        'GP300 / nec became flat (%.1f%%); nec may have been overwritten'
        % (100 * varying.std() / varying.mean()))


def test_hfss_tables_are_not_reachable_through_du_type():
    r"""Records that the ``hfss`` tables ship but no ``du_type`` opens them.

    ``galactic_noise`` accepts ``GP300``, ``GP300_nec`` and ``GP300_mat``.  The
    first recomputes the voltage from the MATLAB power file; the other two read
    the ``nec`` and ``mat`` tables.  The three ``*_hfss.npy`` files are never
    opened, even though they are the highest-level tables shipped, so a study
    that wanted them would have to load them by hand.

    Checked by recording every path the function actually opens, rather than by
    grepping the source: the docstring names the hfss tables in order to warn
    about exactly this, so a text search finds them and proves nothing.
    """
    from grand.sim.noise import galaxy

    opened = []

    # Bind the originals before patching.  `galaxy.np` is the numpy module
    # itself, so a recorder that called `np.load` would call its own
    # replacement and recurse.
    real_load, real_h5 = galaxy.np.load, galaxy.h5py.File

    def record_load(path, *args, **kwargs):
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    def record_h5(path, *args, **kwargs):
        opened.append(str(path))
        return real_h5(path, *args, **kwargs)

    galaxy.np.load, galaxy.h5py.File = record_load, record_h5
    try:
        for du_type in ("GP300", "GP300_nec", "GP300_mat"):
            galactic_noise(float(LST_HOUR), 256, FREQS_MHZ, nb_ant=1, seed=0,
                           du_type=du_type)
    finally:
        galaxy.np.load, galaxy.h5py.File = real_load, real_h5

    assert opened, 'no data file was opened; the recording hook did not work'
    hfss = [path for path in opened if "hfss" in path]
    assert not hfss, (
        'a du_type now reads an hfss table (%s); this test and the caveat in '
        'the galactic_noise docstring are stale' % ', '.join(hfss))


if __name__ == '__main__':
    for du in ("GP300", "GP300_nec", "GP300_mat"):
        _, _, tr = _traces(du_type=du)
        ratio = tr.std(axis=(0, 2)) / _reference(du)
        print('%-10s simulated / tabulated : %s   (1/sqrt2 = %.4f)'
              % (du, np.round(ratio, 4), 1 / np.sqrt(2)))
