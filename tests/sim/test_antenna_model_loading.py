# -*- coding: utf-8 -*-
r"""Covers the antenna-model loader for every ``du_type`` it accepts.

Only the default path was exercised, leaving 50 of ``antenna_model.py``'s 102
statements uncovered -- the ``GP300_nec``, ``GP300_mat`` and ``Horizon``
branches among them.  Those branches choose which measured response the whole
simulation is built on, so a mistake in one is not a crash: it is a different
answer.

Complements ``tests/sim/test_antenna_arm_identity.py``, which establishes
*which physical arm* each file describes.  This file checks that each branch
loads, and that the tables it produces are self-consistent and physically
plausible.
"""

import numpy as np
import pytest

from grand.sim.detector.antenna_model import AntennaModel

#: The options the loader documents.  ``Horizon`` reads a different file
#: format (``.npy`` rather than ``.npz``) and may not be present.
DU_TYPES = ['GP300', 'GP300_nec', 'GP300_mat']

ARMS = ('sn', 'ew', 'z')


@pytest.fixture(scope='module')
def models():
    r"""Returns one loaded model per ``du_type``, built once.

    Returns
    -------
    dict
        Mapping from ``du_type`` to :class:`AntennaModel`.
    """
    return {du: AntennaModel(du_type=du) for du in DU_TYPES}


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_every_du_type_loads_three_arms(models, du_type):
    r"""Each option produces a model with all three arms populated."""
    model = models[du_type]
    assert set(model.d_leff) == set(ARMS), (
        '%s gave arms %s' % (du_type, sorted(model.d_leff)))
    for arm in ARMS:
        table = model.d_leff[arm]
        assert table is not None, '%s: arm %s is None' % (du_type, arm)
        assert getattr(model, 'leff_%s' % arm) is table, (
            '%s: leff_%s and d_leff["%s"] are different objects'
            % (du_type, arm, arm))


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_the_grid_is_the_documented_one(models, du_type):
    r"""30-250 MHz in 221 bins, azimuth 0-360, zenith 0-90, all in degrees.

    The frequency axis is stored in **hertz**, unlike everything in
    :mod:`grand.sim.detector.rf_chain`, which uses megahertz.  Pinned here
    because the attribute name carries no unit and the mismatch is a factor of
    a million.
    """
    for arm in ARMS:
        table = models[du_type].d_leff[arm]
        assert table.frequency.shape == (221,), (
            '%s/%s: %s frequency bins' % (du_type, arm, table.frequency.shape))
        assert np.isclose(table.frequency.min(), 30e6), (
            '%s/%s: frequency axis starts at %g, not 3e7 Hz -- if this is 30, '
            'the tables have moved to MHz and callers dividing by 1e6 are now '
            'wrong' % (du_type, arm, table.frequency.min()))
        assert np.isclose(table.frequency.max(), 250e6)
        assert table.phi.shape == (361,) and table.theta.shape == (91,)
        assert (table.phi.min(), table.phi.max()) == (0.0, 360.0)
        assert (table.theta.min(), table.theta.max()) == (0.0, 90.0)


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_the_response_is_complex_and_finite(models, du_type):
    r"""The tabulated effective length is complex and has no holes.

    Complex because the antenna disperses as well as scales; a table that came
    back real would mean the phase had been dropped somewhere, which would cost
    the sub-nanosecond timing the reconstruction depends on.
    """
    for arm in ARMS:
        table = models[du_type].d_leff[arm]
        for name in ('leff_theta_reim', 'leff_phi_reim'):
            values = getattr(table, name)
            assert values.shape == (221, 361, 91), (
                '%s/%s/%s has shape %s' % (du_type, arm, name, values.shape))
            assert np.iscomplexobj(values), (
                '%s/%s/%s is not complex' % (du_type, arm, name))
            assert np.isfinite(values).all(), (
                '%s/%s/%s contains non-finite values' % (du_type, arm, name))
            assert np.abs(values).max() > 0.0, (
                '%s/%s/%s is all zero' % (du_type, arm, name))


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_the_polar_form_is_never_populated(models, du_type):
    r"""``leff_theta`` and ``phase_theta`` exist and are ``None``.

    A trap rather than a bug: the tables ship in real/imaginary form, and code
    reaching for the polar attributes gets ``None`` rather than an error.
    Asserted so that the day they are populated, the documentation saying they
    are not becomes provably stale.
    """
    for arm in ARMS:
        table = models[du_type].d_leff[arm]
        for name in ('leff_theta', 'leff_phi', 'phase_theta', 'phase_phi'):
            assert getattr(table, name) is None, (
                '%s/%s: %s is now populated; docs and notebook 03 say it is '
                'not' % (du_type, arm, name))


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_the_effective_length_is_physically_plausible(models, du_type):
    r"""Peak magnitudes are metres, not millimetres or kilometres.

    A crude check, and that is the intention: it catches a unit error or a
    transposed axis without pinning numbers that a legitimate re-measurement of
    the antenna would change.
    """
    for arm in ARMS:
        amplitude = np.abs(models[du_type].d_leff[arm].leff_theta_reim)
        peak = amplitude.max()
        assert 0.1 < peak < 20.0, (
            '%s/%s: peak |l_theta| is %.3g m, outside anything plausible for a '
            'few-metre antenna' % (du_type, arm, peak))


@pytest.mark.parametrize('du_type', DU_TYPES)
def test_the_response_vanishes_at_the_horizon(models, du_type):
    r"""Every arm goes to zero at zenith 90 degrees.

    A property of the tabulation rather than of the antenna: the model is not
    defined below the horizontal.  Worth pinning because a reconstruction that
    accepted events at 90 degrees would be dividing by it.
    """
    for arm in ARMS:
        amplitude = np.abs(models[du_type].d_leff[arm].leff_theta_reim)
        assert np.allclose(amplitude[:, :, -1], 0.0), (
            '%s/%s: the response at the horizon is not zero (max %.3g)'
            % (du_type, arm, amplitude[:, :, -1].max()))


def test_the_z_arm_is_not_a_scaled_copy_of_the_horizontal_arms(models):
    r"""The vertical arm has a different band shape, not a different scale.

    The horizontal arms peak near 142 MHz and the Z arm near 49; an analysis
    assuming the three channels share a band shape is assuming something false.
    Checked as an inequality of peak frequencies so it survives a
    re-measurement that moved either.
    """
    model = models['GP300']
    freq = model.leff_sn.frequency
    peaks = {}
    for arm in ARMS:
        amplitude = np.abs(model.d_leff[arm].leff_theta_reim)
        index = np.unravel_index(amplitude.argmax(), amplitude.shape)[0]
        peaks[arm] = freq[index]

    assert abs(peaks['sn'] - peaks['ew']) < 20e6, (
        'the two horizontal arms peak far apart: %s' % peaks)
    assert abs(peaks['z'] - peaks['sn']) > 20e6, (
        'the Z arm now peaks with the horizontal arms: %s -- the documentation '
        'and notebook 03 say it does not' % peaks)


def test_an_unknown_du_type_does_not_silently_give_an_empty_model():
    r"""Asking for a model that does not exist must not return a usable object.

    The loader is a chain of ``if``/``elif`` with no ``else``, so an unknown
    name falls through.  This records what actually happens, which is that the
    arms are never assigned and attribute access fails -- unhelpful, but not
    silent.
    """
    with pytest.raises((AttributeError, KeyError, UnboundLocalError, NameError)):
        model = AntennaModel(du_type='no_such_antenna')
        _ = model.d_leff['sn'].frequency
