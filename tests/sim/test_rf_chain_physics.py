# -*- coding: utf-8 -*-
r"""Physical checks on the RF chain, rather than "did it run" checks.

``tests/sim/test_rf_chain.py`` has 89 assertions and nearly all of them are
``assertEqual(np.sum(x), 0)`` before a computation and ``assertNotEqual``
after.  That establishes only that something was written.  A sign error, a
factor of two, a swapped port or a mis-ordered cascade all pass it.

The properties below are ones a correct chain has and an incorrect one
generally does not, so they fail for a reason rather than by coincidence.
"""

import numpy as np
import pytest

from grand.sim.detector.rf_chain import RFChain, db2reim, matmul, s2abcd

FREQS_MHZ = np.arange(30.0, 251.0)


def test_ideal_through_line_is_the_identity():
    r"""A matched, lossless two-port has the identity ABCD matrix.

    :math:`S_{11}=S_{22}=0` and :math:`S_{12}=S_{21}=1` describe a piece of
    perfect wire.  Cascading it must change nothing, which is exactly the
    statement that its ABCD matrix is the identity.
    """
    n = 8
    one, zero = np.ones(n, dtype=complex), np.zeros(n, dtype=complex)
    abcd = s2abcd(zero, one, one, zero)
    assert abcd.shape == (2, 2, n)
    assert np.allclose(abcd, np.eye(2)[:, :, None]), (
        'the ideal through-line is not the identity')


def test_cascading_with_the_identity_changes_nothing():
    r"""Multiplying by the identity leaves a stage untouched.

    The property that makes the ABCD representation worth converting to: the
    cascade of two networks is the product of their matrices, so the identity
    must be a unit for it.
    """
    rng = np.random.default_rng(0)
    n = 6
    a = (rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n)))
    identity = np.zeros((2, 2, n), dtype=complex)
    identity[0, 0] = identity[1, 1] = 1.0

    assert np.allclose(matmul(a, identity), a), 'right identity failed'
    assert np.allclose(matmul(identity, a), a), 'left identity failed'


def test_matmul_agrees_with_numpy():
    r"""The hand-written 2x2 product matches :func:`numpy.matmul`.

    ``matmul`` here multiplies over stacked trailing axes, which is easy to
    get subtly wrong -- transposed, or contracting the wrong index.  NumPy is
    the independent implementation to check it against.
    """
    rng = np.random.default_rng(1)
    n = 5
    a = rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n))
    b = rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n))

    expected = np.einsum('ikn,kjn->ijn', a, b)
    assert np.allclose(matmul(a, b), expected), (
        'the 2x2 cascade does not agree with an explicit contraction')


def test_matmul_is_not_commutative():
    r"""Cascade order matters, and the implementation respects it.

    If ``matmul`` were symmetric in its arguments, the chain would give the
    same answer with its stages in any order -- which would mean the ordering
    in :class:`RFChain` was not doing anything.
    """
    rng = np.random.default_rng(2)
    n = 4
    a = rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n))
    b = rng.normal(size=(2, 2, n)) + 1j * rng.normal(size=(2, 2, n))
    assert not np.allclose(matmul(a, b), matmul(b, a))


def test_db2reim_uses_the_voltage_convention():
    r"""Decibels are converted as a voltage ratio, not a power ratio.

    :math:`|z| = 10^{dB/20}`, so -20 dB is a factor of ten in amplitude and
    not a factor of a hundred.  The S-parameters this module reads are
    measured as voltage ratios by a vector network analyser, so the power
    convention would be wrong by a square.
    """
    re, im = db2reim(np.array([0.0, -20.0, 20.0]), np.zeros(3))
    assert np.isclose(re[0], 1.0), '0 dB is not unit modulus'
    assert np.isclose(re[1], 0.1), '-20 dB is not a factor of ten'
    assert np.isclose(re[2], 10.0), '+20 dB is not a factor of ten'
    assert np.allclose(im, 0.0), 'zero phase gave an imaginary part'


def test_db2reim_phase_is_in_radians():
    r"""A quarter turn of phase lands on the imaginary axis."""
    re, im = db2reim(np.array([0.0]), np.array([np.pi / 2]))
    assert np.isclose(re[0], 0.0, atol=1e-12)
    assert np.isclose(im[0], 1.0)


@pytest.fixture(scope='module')
def chain():
    r"""Returns an RF chain evaluated over the 30-250 MHz band."""
    rf = RFChain(vga_gain=20)
    rf.compute_for_freqs(FREQS_MHZ)
    return rf


def test_transfer_function_shape_and_finiteness(chain):
    r"""The chain gives one finite transfer function per antenna arm."""
    tf = chain.get_tf()
    assert tf.shape[0] == 3, 'expected one transfer function per arm'
    assert tf.shape[1] == FREQS_MHZ.size
    assert np.isfinite(tf).all(), 'the transfer function is not finite'


def test_the_chain_amplifies_in_band(chain):
    r"""With a 20 dB amplifier the chain is a net amplifier.

    Figure 8 of the GRANDlib paper shows :math:`|V_{\rm out}/V_{\rm oc}|`
    peaking well above unity across the band for this gain setting.
    """
    gain = np.abs(chain.get_tf())
    assert gain.max() > 1.0, 'a 20 dB chain did not amplify at any frequency'
    assert gain.max() < 1e4, (
        'implausible gain %.3g; a factor slipped somewhere' % gain.max())


@pytest.mark.xfail(reason='vga_gain is currently ignored: the line selecting '
                          'the per-gain S-parameter file is commented out in '
                          'VGAFilter._set_name_data_file. See known_issues.',
                   strict=False)
def test_gain_setting_changes_the_transfer_function():
    r"""A different VGA gain gives a different answer.

    Guards against the gain argument being ignored, which would make every
    detector-design comparison that varies it meaningless.
    """
    low, high = RFChain(vga_gain=0), RFChain(vga_gain=20)
    low.compute_for_freqs(FREQS_MHZ)
    high.compute_for_freqs(FREQS_MHZ)
    assert not np.allclose(np.abs(low.get_tf()), np.abs(high.get_tf())), (
        'vga_gain made no difference to the transfer function')
    assert np.abs(high.get_tf()).max() > np.abs(low.get_tf()).max(), (
        '20 dB did not give more gain than 0 dB')


def test_z_arm_differs_from_the_horizontal_arms(chain):
    r"""The vertical arm has its own response.

    Section 8.3 of the paper notes that the Z port's chain differs from the
    X and Y ports.  Identical transfer functions would mean the distinction
    had been lost.
    """
    tf = np.abs(chain.get_tf())
    assert not np.allclose(tf[2], tf[0]), 'the Z arm matches the X arm exactly'
