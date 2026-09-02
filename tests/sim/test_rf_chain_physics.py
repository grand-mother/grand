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


# --------------------------------------------------------------------------
# Passivity and reciprocity
#
# The tests above check the algebra of the cascade.  These check that the
# measured data going into it describes physically possible components, which
# is a different question and the one that catches a corrupted file, a swapped
# column or a magnitude read in the wrong units.
#
# Two properties, and the point of both is that they *discriminate*: they must
# hold for the passive stages and must fail for the amplifiers.  A test that
# passed for everything would be measuring nothing.
# --------------------------------------------------------------------------

#: Stages with no power source.  A passive two-port cannot emit more power than
#: it receives, and is reciprocal up to measurement error.
PASSIVE = ('matcnet', 'balun1', 'cable', 'balun2')

#: Stages that amplify.  `vgaf` is in this list because of what it actually
#: loads -- feb+amfitler+biast.s2p, a front-end board -- and not because a
#: filter should have gain; see the vga_gain entry in the known issues.
ACTIVE = ('lna', 'vgaf')

#: Reciprocity tolerance for measured data.  The measured stages sit at
#: 1.2e-3 to 1.1e-2; the matching network is simulated and is exact.  The
#: amplifiers are three orders of magnitude above this.
RECIPROCITY_TOL = 5e-2


def _sparams(chain, name):
    r"""Returns ``(s11, s12, s21)`` for one stage as complex arrays.

    Parameters
    ----------
    chain : RFChain
        A chain on which ``compute_for_freqs`` has been called.
    name : str
        Attribute name of the stage.

    Returns
    -------
    tuple of ndarray
        Each of shape ``(3, n_freq)``.
    """
    stage = getattr(chain, name)
    return (np.asarray(stage.s11), np.asarray(stage.s12), np.asarray(stage.s21))


@pytest.mark.parametrize('name', PASSIVE)
def test_passive_stages_do_not_create_power(chain, name):
    r"""A passive two-port cannot emit more power than it receives.

    For a lossless-or-lossy passive network driven at port 1,
    :math:`|S_{11}|^2 + |S_{21}|^2 \le 1`: what is not reflected or transmitted
    was dissipated.  Exceeding one means the component is generating power,
    which for a balun or a length of coaxial cable would mean the data is wrong
    -- most plausibly a magnitude read in the wrong units, since these files
    come in two different Touchstone formats.
    """
    s11, _, s21 = _sparams(chain, name)
    delivered = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert delivered.max() <= 1.0 + 1e-6, (
        '%s emits more power than it receives: max |S11|^2 + |S21|^2 = %.4f'
        % (name, delivered.max()))


@pytest.mark.parametrize('name', ACTIVE)
def test_active_stages_do_create_power(chain, name):
    r"""The amplifiers must fail the passivity test, or it is measuring nothing.

    The guard against a passivity check that would pass on any input at all --
    for instance if every magnitude were being read as a small number.
    """
    s11, _, s21 = _sparams(chain, name)
    delivered = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert delivered.max() > 1.0, (
        '%s does not amplify: max |S11|^2 + |S21|^2 = %.4f. Either the stage '
        'is loading the wrong file or a dB magnitude is being read as linear.'
        % (name, delivered.max()))


@pytest.mark.parametrize('name', PASSIVE)
def test_passive_stages_are_reciprocal(chain, name):
    r"""A passive network transmits equally in both directions: :math:`S_{12} = S_{21}`.

    True of any network built from reciprocal materials, which is everything in
    this chain except the amplifiers.  It is a genuine check on the data rather
    than on the loader: ``s12`` is read from its own columns of the Touchstone
    file, not copied from ``s21``, so agreement is a property of the
    measurement.

    The matching network is exact -- it is simulated rather than measured --
    while the measured stages agree to about 1 %.
    """
    _, s12, s21 = _sparams(chain, name)
    difference = np.abs(s12 - s21).max()
    assert difference < RECIPROCITY_TOL, (
        '%s is not reciprocal: max |S12 - S21| = %.3e, above the %.0e expected '
        'of measurement error' % (name, difference, RECIPROCITY_TOL))


@pytest.mark.parametrize('name', ACTIVE)
def test_active_stages_are_not_reciprocal(chain, name):
    r"""An amplifier passes signal one way, so :math:`S_{12} \ne S_{21}`.

    The counterpart to the test above: without this, a bug that set ``s12 =
    s21`` for every stage would leave the reciprocity test passing everywhere
    and looking like a strong result.
    """
    _, s12, s21 = _sparams(chain, name)
    difference = np.abs(s12 - s21).max()
    assert difference > RECIPROCITY_TOL, (
        '%s looks reciprocal (max |S12 - S21| = %.3e); an amplifier should not '
        'be, so s12 may be a copy of s21' % (name, difference))


def test_the_cascade_is_lossier_than_its_amplifiers_alone(chain):
    r"""End-to-end gain is below the product of the two amplifier gains.

    A weak statement deliberately: it does not pin a number that a
    re-measurement of any component would change, but it does catch a cascade
    that has silently dropped its passive stages -- which would leave the
    matched, lossy parts out and inflate the transfer function.
    """
    gain = np.abs(chain.get_tf()).max()
    amplifier_product = 1.0
    for name in ACTIVE:
        _, _, s21 = _sparams(chain, name)
        amplifier_product *= np.abs(s21).max()

    assert gain < amplifier_product, (
        'the whole chain (%.1f) is not lossier than its amplifiers alone '
        '(%.1f); the passive stages may not be in the cascade'
        % (gain, amplifier_product))
