# -*- coding: utf-8 -*-
r"""Covers the peak interpolators in :mod:`grand.basis.signal`.

``find_max_with_parabola_interp`` was entirely untested -- 50 of the module's
93 statements, the single largest uncovered block in ``grand.basis``.  It is
what turns a sampled peak into a sub-sample arrival time, so an error in it
becomes a timing error in every reconstruction downstream, and one that would
look like jitter rather than like a bug.

The tests use an exact parabola, because then the right answer is known
analytically rather than being whatever the function returned when the test was
written.
"""

import numpy as np
import pytest

from grand.basis.signal import (find_max_with_parabola_interp,
                                find_max_with_parabola_interp_3pt)

#: The parabola the tests fit: peak of 40 at x = 1.
X_TRUE, Y_TRUE, CURVATURE = 1.0, 40.0, -10.0


def parabola(x):
    r"""Returns the reference parabola evaluated at `x`."""
    return CURVATURE * (x - X_TRUE) ** 2 + Y_TRUE


def sampled(n=201, span=3.0):
    r"""Returns a sampling of the reference parabola and the index of its peak.

    Parameters
    ----------
    n : int, optional
        Number of samples.
    span : float, optional
        Range of `x`.

    Returns
    -------
    x, y : ndarray
        The samples.
    idx : int
        Index of the largest sample.
    """
    x = np.linspace(0.0, span, n)
    y = parabola(x)
    return x, y, int(np.argmax(y))


@pytest.mark.parametrize('interp', [find_max_with_parabola_interp_3pt,
                                    find_max_with_parabola_interp])
@pytest.mark.parametrize('n', [61, 201, 512])
def test_recovers_the_true_peak_of_a_parabola(interp, n):
    r"""Both interpolators recover the analytic maximum, not the sampled one.

    That distinction is the point of the function: at 201 samples over this
    span the largest *sample* is at x = 1.005, half a bin away from the true
    peak, and returning it would be a timing error of half a sample.
    """
    x, y, idx = sampled(n=n)
    x_max, y_max = interp(x, y, idx)

    assert np.isclose(x_max, X_TRUE, atol=1e-6), (
        '%s put the peak at %.6f, not %.1f' % (interp.__name__, x_max, X_TRUE))
    assert np.isclose(y_max, Y_TRUE, atol=1e-6), (
        '%s gave a peak value of %.6f, not %.1f' % (interp.__name__, y_max, Y_TRUE))


def test_interpolation_beats_taking_the_largest_sample():
    r"""The interpolated position is closer to the truth than the sample is.

    Asserted as an inequality rather than a tolerance, so it stays meaningful
    whatever the sampling: an interpolator that were no better than ``argmax``
    would have no reason to exist.
    """
    x, y, idx = sampled(n=201)
    naive = abs(x[idx] - X_TRUE)
    refined = abs(find_max_with_parabola_interp(x, y, idx)[0] - X_TRUE)

    assert naive > 1e-3, 'the sampling is too fine for this test to mean anything'
    assert refined < naive / 10.0, (
        'interpolation gave %.2e against %.2e for the largest sample'
        % (refined, naive))


def test_the_two_interpolators_agree_on_a_clean_peak():
    r"""The three-point and hill fits coincide where both are valid.

    They take different amounts of the peak -- three samples against however
    many lie above ``factor_hill`` of it -- so agreeing on an exact parabola is
    a check that the hill branch's overdetermined solve is set up correctly.
    """
    x, y, idx = sampled(n=201)
    a = find_max_with_parabola_interp_3pt(x, y, idx)
    b = find_max_with_parabola_interp(x, y, idx)
    assert np.allclose(a, b, atol=1e-6), '3pt gave %s, hill gave %s' % (a, b)


@pytest.mark.parametrize('factor_hill', [0.5, 0.8, 0.96, 0.99])
def test_the_hill_fraction_does_not_change_the_answer_on_a_parabola(factor_hill):
    r"""``factor_hill`` selects how much of the peak to fit, not what it is.

    On an exact parabola every fraction fits the same curve, so the result must
    not move.  On a real pulse it would, which is why the parameter exists;
    this pins that it is a choice of window and not a fudge factor.
    """
    x, y, idx = sampled(n=201)
    x_max, y_max = find_max_with_parabola_interp(x, y, idx, factor_hill=factor_hill)
    assert np.isclose(x_max, X_TRUE, atol=1e-6), (
        'factor_hill=%.2f moved the peak to %.6f' % (factor_hill, x_max))
    assert np.isclose(y_max, Y_TRUE, atol=1e-6)


def test_falls_back_to_three_points_on_a_narrow_spike():
    r"""A one-sample spike leaves too little hill, and the fit still returns.

    The branch that matters: with fewer than three samples above the threshold
    the function must take the three-point path rather than trying to solve an
    underdetermined system.  The value is not meaningful for a spike -- there is
    no parabola to find -- so this checks only that it is finite and near the
    spike.
    """
    x = np.linspace(0.0, 3.0, 201)
    y = np.zeros_like(x)
    y[99], y[100], y[101] = 3.0, 10.0, 4.0

    x_max, y_max = find_max_with_parabola_interp(x, y, 100)
    assert np.isfinite(x_max) and np.isfinite(y_max), 'the fit returned non-finite values'
    assert abs(x_max - x[100]) < (x[1] - x[0]) * 2, (
        'the peak moved %.4f from the spike, more than two samples'
        % abs(x_max - x[100]))
    assert y_max >= y[100], 'the interpolated peak is below the largest sample'


def test_a_shifted_peak_moves_the_answer_with_it():
    r"""Translating the trace translates the result by the same amount.

    A guard against an off-by-one in the index handling, which would show up
    here as a constant offset rather than as an obviously wrong number.
    """
    x = np.linspace(0.0, 3.0, 201)
    for shift in (-0.25, 0.0, 0.25, 0.5):
        y = CURVATURE * (x - (X_TRUE + shift)) ** 2 + Y_TRUE
        idx = int(np.argmax(y))
        x_max, _ = find_max_with_parabola_interp(x, y, idx)
        assert np.isclose(x_max, X_TRUE + shift, atol=1e-6), (
            'peak at %.3f was found at %.6f' % (X_TRUE + shift, x_max))
