# Created by Sebastian Castro-Isern on 20/01/2026
r"""Cramér-Rao lower bounds on the parameters of the reconstruction fits.

Each bound is the square root of a diagonal element of the inverse Fisher
information matrix, built from numerical derivatives of the fit models at the
fitted parameters and the assumed measurement uncertainties.  It is the
smallest standard deviation an unbiased fit can reach, so it gives the order
of magnitude of the reconstruction errors.
"""

import numpy                              as np 
import grand.analysis.fitting.adf         as adf
import grand.analysis.fitting.spherical   as swf
import grand.analysis.fitting.plane_wave  as pwf
from grand.analysis import _checks
from grand.basis import validate as _validate

#: Above this condition number of the unit-diagonal Fisher matrix, the
#: measurements do not constrain the parameters and the bounds mean nothing.
_ILL_CONDITIONED = 1e10


def _warn_if_ill_conditioned(fisher, where, threshold=_ILL_CONDITIONED):
    r"""Warns when the Fisher matrix is close to singular.

    The matrix is first scaled to unit diagonal, so that parameters in very
    different units (radians, metres, seconds) do not make every case look
    ill-conditioned.
    """
    diag = np.sqrt(np.abs(np.diag(fisher)))
    if np.any(diag == 0):
        _validate.warn(where, "a parameter has no effect on the measurements, so its bound is infinite", stacklevel=4)
        return
    cond = np.linalg.cond(fisher / np.outer(diag, diag))
    if not np.isfinite(cond) or cond > threshold:
        _validate.warn(where, "the antennas barely constrain the parameters (condition number %.2g); "
                       "the bounds are not meaningful. Use more antennas, or antennas not on a line" % cond,
                       stacklevel=4)

def CRB_ADF_SWF(theta_swf: float, phi_swf: float, r_xsource: float, t_s: float, theta_adf: float, phi_adf: float, delta_omega: float, scaling_factor: float, Xants: np.ndarray, uncertainty_amplitude= 0.075, uncertainty_time=5e-9) -> np.ndarray:
    """
    Cramer-Rao Bound (CRB) calculation for the joint ADF (amplitudes) and SWF (times) model.

    Parameters
    ----------
    theta_swf : float
        Zenith angle from SWF reconstruction (radians).
    phi_swf : float
        Azimuth angle from SWF reconstruction (radians).
    r_xsource : float
        Distance to the source (meters).
    t_s : float
        Emission time of the source (seconds).
    theta_adf : float
        Zenith angle from ADF reconstruction (radians).
    phi_adf : float
        Azimuth angle from ADF reconstruction (radians).
    delta_omega : float
        Angular uncertainty (radians).
    scaling_factor : float
        Scaling factor for the amplitude model.
    Xants : np.ndarray
        Antenna positions, shape (N, 3).
    uncertainty_amplitude : float, optional
        Relative uncertainty in amplitude measurements (current 7.5%).
    uncertainty_time : float, optional
        Uncertainty in time measurements (seconds).

    Returns
    -------
    np.ndarray, shape (8,)
        The smallest standard deviations (square roots of the diagonal of the
        inverse Fisher matrix) an unbiased fit can reach, for the parameters
        in the order of the arguments: theta_swf, phi_swf (radians),
        r_xsource (m), t_s (s), theta_adf, phi_adf, delta_omega (radians),
        scaling_factor.  All NaN if the Fisher matrix cannot be inverted
        (#261).
    """
    # Number of antennas
    where = "CRB_ADF_SWF"
    Xants = _checks.antennas(Xants, where, min_ants=4)
    for _name, _value in (("theta_swf", theta_swf), ("phi_swf", phi_swf), ("r_xsource", r_xsource),
                          ("t_s", t_s), ("theta_adf", theta_adf), ("phi_adf", phi_adf),
                          ("delta_omega", delta_omega), ("scaling_factor", scaling_factor)):
        _validate.as_real(_value, _name, where)
    _validate.positive(_validate.as_real(uncertainty_amplitude, "uncertainty_amplitude", where), "uncertainty_amplitude", where)
    _validate.positive(_validate.as_real(uncertainty_time, "uncertainty_time", where), "uncertainty_time", where, "s")
    nants = Xants.shape[0]
    params = [theta_swf, phi_swf, r_xsource, t_s, theta_adf, phi_adf, delta_omega, scaling_factor]
    
    # Allocate memory for Fisher Information Matrix
    fisher_information_matrix = np.zeros((8, 8))
    derivates_ampl = np.zeros((Xants.shape[0], 8))
    derivates_time = np.zeros((Xants.shape[0], 8))

    # Parameters array, and step sizes for numerical derivatives
    params = np.hstack([params])
    h = 1e-6 * np.abs(params)
    h[3] = 1e-9  # bigger step for time to avoid numerical issues
    h[h == 0] = 1e-6  # a parameter at exactly 0 (e.g. azimuth) would get no step, and 0/0

    # Derivate on each antenna for each parameter
    for i in range(8):
        params_plus  = params.copy()
        params_plus[i]  += h[i]
        params_minus = params.copy()
        params_minus[i] -= h[i]

        # Calculate Xsource for amplitude model with perturbed parameters (plus)
        Xsource_plus = swf.compute_Xsource_cartesian_coords(params_plus[0], params_plus[1], params_plus[2])
        Xsource_plus = np.asarray(Xsource_plus)[0]

        # Calculate Xsource for amplitude model with perturbed parameters (minus)
        Xsource_minus = swf.compute_Xsource_cartesian_coords(params_minus[0], params_minus[1], params_minus[2])
        Xsource_minus = np.asarray(Xsource_minus)[0]

        # Extract ADF parameters (plus)
        theta_plus, phi_plus, delta_omega_plus, scaling_factor_plus = params_plus[4], params_plus[5], params_plus[6], params_plus[7]

        # Extract ADF parameters (minus)
        theta_minus, phi_minus, delta_omega_minus, scaling_factor_minus = params_minus[4], params_minus[5], params_minus[6], params_minus[7]

        # Derivate over amplitude
        _,_,_,_, pred_ampl_plus = adf.ADF_parameters(theta_plus, phi_plus, delta_omega_plus, scaling_factor_plus, Xants, Xsource_plus)
        _,_,_,_, pred_ampl_minus = adf.ADF_parameters(theta_minus, phi_minus, delta_omega_minus, scaling_factor_minus, Xants, Xsource_minus)
    
        derivates_ampl[:, i] = (pred_ampl_plus - pred_ampl_minus) / (2.0 * h[i])

        # Derivate over time
        pred_time_plus = swf.SWF_model(params_plus[0], params_plus[1], params_plus[2], params_plus[3], Xants)
        pred_time_minus = swf.SWF_model(params_minus[0], params_minus[1], params_minus[2], params_minus[3], Xants)

        derivates_time[:, i] = (pred_time_plus - pred_time_minus) / (2.0 * h[i])

    # Get predicted amplitudes for sigma calculation
    Xsource = swf.compute_Xsource_cartesian_coords(theta_swf, phi_swf, r_xsource)
    Xsource = np.asarray(Xsource)[0]
    _,_,_,_, pred_ampl = adf.ADF_parameters(theta_adf, phi_adf, delta_omega, scaling_factor, Xants, Xsource)

    # Fill Fisher Information Matrix
    sigma_ampl = (uncertainty_amplitude * pred_ampl) # add galactic noise later if known
    sigma_time = (uncertainty_time) # in seconds

    for i in range(nants):
        fisher_information_matrix += np.outer(derivates_ampl[i, :], derivates_ampl[i, :]) / (sigma_ampl[i] ** 2)
        fisher_information_matrix += np.outer(derivates_time[i, :], derivates_time[i, :]) / (sigma_time ** 2)
    
    # Its eight parameters are strongly correlated by construction, so only a
    # singular matrix is flagged here.
    _warn_if_ill_conditioned(fisher_information_matrix, where, threshold=1e16)
    try:
        cov_matrix = np.linalg.inv(fisher_information_matrix)
        crb_values = np.sqrt(np.diag(cov_matrix))
        if np.any(np.isnan(crb_values)):
            print("Cramer-Rao Bound computation resulted in NaN values.")
        if np.any(np.isinf(crb_values)):
            print("Cramer-Rao Bound computation resulted in Inf values.")
        return crb_values
    except np.linalg.LinAlgError:
        print("Fisher Information Matrix is singular, cannot compute CRB on ADF and SWF.")
        return np.full(8, np.nan)
    
def CRB_PWF(theta_pwf: float, phi_pwf: float, Xants: np.ndarray, uncertainty_time=5e-9) -> np.ndarray:
    """
    Cramer-Rao Bound (CRB) calculation for the Plane Wave Fit (PWF) model.

    Parameters
    ----------
    theta_pwf : float
        Polar angle from PWF reconstruction (radians).
    phi_pwf : float
        Azimuth angle from PWF reconstruction (radians).
    Xants : np.ndarray
        Antenna positions, shape (N, 3).
    uncertainty_time : float, optional
        Uncertainty in time measurements (seconds).

    Returns
    -------
    np.ndarray, shape (2,)
        The smallest standard deviations of theta_pwf and phi_pwf an unbiased
        fit can reach, in radians (square roots of the diagonal of the
        inverse Fisher matrix) (#261).
    """
    # Number of antennas
    where = "CRB_PWF"
    Xants = _checks.antennas(Xants, where, min_ants=3)
    _checks.angles(where, theta_pwf=theta_pwf, phi_pwf=phi_pwf)
    _validate.positive(_validate.as_real(uncertainty_time, "uncertainty_time", where), "uncertainty_time", where, "s")
    nants = Xants.shape[0]

    # Allocate memory for Fisher Information Matrix
    fisher_information_matrix = np.zeros((2, 2))
    derivates_time = np.zeros((Xants.shape[0], 2))

    # Parameters array, and step sizes for numerical derivatives
    params = np.array([theta_pwf, phi_pwf])
    h = 1e-6 * np.abs(params)
    h[h == 0] = 1e-6  # a parameter at exactly 0 (e.g. azimuth) would get no step, and 0/0

    # Derivate on each antenna for each parameter
    for i in range(2):
        params_plus  = params.copy()
        params_plus[i]  += h[i]
        params_minus = params.copy()
        params_minus[i] -= h[i]

        pred_time_plus  = pwf.PWF_model(params_plus, Xants)
        pred_time_minus = pwf.PWF_model(params_minus, Xants)

        derivates_time[:, i] = (pred_time_plus - pred_time_minus) / (2.0 * h[i])

    # Fill Fisher Information Matrix
    sigma_time = (uncertainty_time) # in seconds

    for i in range(nants):
        fisher_information_matrix += np.outer(derivates_time[i, :], derivates_time[i, :]) / (sigma_time ** 2)
    
    _warn_if_ill_conditioned(fisher_information_matrix, where)
    try:
        cov_matrix = np.linalg.inv(fisher_information_matrix)
        crb_values = np.sqrt(np.diag(cov_matrix))
        return crb_values
    except np.linalg.LinAlgError:
        print("Fisher Information Matrix is singular, cannot compute CRB on PWF.")
        return np.full(2, np.nan)
