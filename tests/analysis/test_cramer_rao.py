# -*- coding: utf-8 -*-
r"""The Cramér-Rao bounds are the spread the fits actually show.

``grand/analysis/cramer_rao_bounds`` (Sebastián Castro-Isern, PR 150) came
without tests.  The strongest check available without data: a Cramér-Rao
bound is the smallest standard deviation an unbiased estimator can reach, and
the plane-wave fit is close to linear, so refitting noisy copies of the same
times must scatter by the bound.  The remaining tests pin the properties the
bound must have whatever the model, and the reading of files written before
the bounds were stored.
"""

import numpy as np
import pytest

pytest.importorskip('iminuit', reason='grand.analysis needs iminuit')

#: Directions (zenith, azimuth) in degrees across the range GRAND observes.
DIRECTIONS = [(35.0, 20.0), (70.0, 137.0), (85.0, 330.0)]

#: The timing uncertainty the bounds assume by default, in seconds.
SIGMA_T = 5e-9


@pytest.fixture(scope='module')
def antennas():
    r"""Twelve antennas scattered over 6 km, near the reference altitude.

    Returns
    -------
    numpy.ndarray
        Positions in metres, shape (12, 3).
    """
    import grand.analysis.constants as cons

    rng = np.random.default_rng(0)
    n = 12
    return np.column_stack([rng.uniform(-3000, 3000, n),
                            rng.uniform(-3000, 3000, n),
                            cons.groundAltitude + rng.uniform(-20, 20, n)])


def _swf_adf(zenith, azimuth):
    r"""A plausible set of SWF and ADF parameters for one direction."""
    theta, phi = np.deg2rad([zenith, azimuth])
    return dict(theta_swf=theta, phi_swf=phi, r_xsource=20000.0, t_s=0.0,
                theta_adf=theta, phi_adf=phi, delta_omega=np.deg2rad(1.0),
                scaling_factor=1e3)


@pytest.mark.parametrize('zenith, azimuth', DIRECTIONS)
def test_the_pwf_bound_is_the_scatter_of_the_plane_wave_fit(antennas, zenith,
                                                           azimuth):
    r"""Noisy times refitted 1000 times scatter by the bound, within 10 %."""
    from grand.analysis.cramer_rao_bounds import CRB_PWF
    from grand.analysis.fitting.plane_wave import PWF_model, PWF_semianalytical

    truth = np.deg2rad([zenith, azimuth])
    times = PWF_model(truth, antennas)
    rng = np.random.default_rng(1)
    fits = np.array([PWF_semianalytical(antennas, times + rng.normal(0, SIGMA_T, len(times)))
                     for _ in range(1000)])

    bound = CRB_PWF(*truth, antennas, uncertainty_time=SIGMA_T)
    assert fits.std(axis=0) == pytest.approx(bound, rel=0.1)


def test_the_bounds_scale_with_the_measurement_uncertainty(antennas):
    r"""Twice the uncertainty on every measurement, twice every bound."""
    from grand.analysis.cramer_rao_bounds import CRB_ADF_SWF, CRB_PWF

    theta, phi = np.deg2rad([70.0, 137.0])
    assert (CRB_PWF(theta, phi, antennas, uncertainty_time=2 * SIGMA_T)
            == pytest.approx(2 * CRB_PWF(theta, phi, antennas), rel=1e-6))

    params = _swf_adf(70.0, 137.0)
    single = CRB_ADF_SWF(**params, Xants=antennas)
    double = CRB_ADF_SWF(**params, Xants=antennas, uncertainty_amplitude=0.15,
                         uncertainty_time=2 * SIGMA_T)
    assert np.all(np.isfinite(single)) and np.all(single > 0)
    assert double == pytest.approx(2 * single, rel=1e-6)


def test_an_azimuth_of_zero_gives_a_finite_bound(antennas):
    r"""A parameter at exactly 0 used to get a derivative step of 0.

    The step was 1e-6 times the parameter, so an azimuth of 0 divided 0 by 0
    and every bound came back NaN.  It now matches the bound just beside it.
    """
    from grand.analysis.cramer_rao_bounds import CRB_ADF_SWF, CRB_PWF

    theta = np.deg2rad(70.0)
    at_zero = CRB_PWF(theta, 0.0, antennas)
    beside = CRB_PWF(theta, 1e-4, antennas)
    assert np.all(np.isfinite(at_zero))
    assert at_zero == pytest.approx(beside, rel=1e-3)

    assert np.all(np.isfinite(CRB_ADF_SWF(**_swf_adf(70.0, 0.0), Xants=antennas)))


def test_the_bounds_are_stored_and_older_files_still_read(tmp_path):
    r"""``TRecons`` writes the bounds; a file from before them still opens.

    The file in ``examples/analysis`` predates the bounds.  Its other fields
    read as before, and the absent bounds read as 0.
    """
    import pathlib
    import shutil

    from grand.dataio import TRecons

    written = tmp_path / 'recons.root'
    tree = TRecons()
    tree.run_number, tree.event_number = 1, 7
    tree.zenith_pwf, tree.crb_zenith_pwf, tree.crb_t_s = 1.2, 0.01, 3e-9
    tree.fill()
    tree.write(str(written))

    back = TRecons(str(written))
    back.get_event(7, 1)
    assert (back.zenith_pwf, back.crb_zenith_pwf, back.crb_t_s) == pytest.approx(
        (1.2, 0.01, 3e-9))

    old = tmp_path / 'old.root'
    shutil.copy(pathlib.Path(__file__).resolve().parents[2] / 'examples'
                / 'analysis' / 'recons_CR_candidates.root', old)
    before = TRecons(str(old))
    before.get_event(479, 10126)
    assert before.zenith_pwf == pytest.approx(1.3426813)
    assert before.crb_zenith_pwf == 0.0
