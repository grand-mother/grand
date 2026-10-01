"""Angular distribution function (ADF) fit of the antenna amplitudes."""

import logging

import numpy as np
import grand.analysis.constants as cons
import grand.analysis.physics as che
import grand.analysis.coords.array_shower as co
import grand.analysis.geom.angles as an
#print(sys.path)
try:
    from iminuit import minimize
except ImportError as _error:          # a bare ModuleNotFoundError named no remedy (#280)
    raise ImportError("GRANDlib: grand.analysis needs the optional package iminuit: "
                      "pip install -e \".[analysis]\" (it is in the conda environment)") from _error
from grand.analysis import _checks
from grand.basis import validate as _validate

logger = logging.getLogger(__name__)

def ADF_parameters(theta, phi, delta_omega, amplitude, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=None):
    """
    Compute all geometric parameters for the ADF function.
    
    Inputs:
        theta, phi          : shower direction angles (rad) (from ADF_recons, best fit values)
        delta_omega         : ADF shape parameter (output from ADF_recons, best fit values)
        amplitude           : ADF amplitude (output from ADF_recons, best fit values)
        Xants  : (N,3) positions of antennas: x North, y West, z above sea level (m)
        Xsource   : (3,) position of Xsource (from SWF)
        Bvec   : (3,) magnetic field
        groundAltitude : height above sea level of the frame's origin (m): 1231 m (GP13) by default; for simulation files, the ground altitude grand.analysis.geom.antenna_positions_from_run returns (#252)
    
    Returns
    -------
        eta       : (N,) azimuthal angle in shower plane
        omega     : (N,) angle wrt shower axis
        omega_cr  : (N,) Cherenkov angle for each antenna
        l_ant     : (N,) distance from Xsource to antenna
        adf       : (N,) ADF amplitude for each antenna
    """
    if Bvec is None:
        Bvec = cons.Bvec
    K = co.shower_direction_vector(theta, phi)
   
    asym_coeff = -0.003*np.rad2deg(theta)+0.220
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2)
    

    l_ant = an.distance_source_antenna(Xants, Xsource)
    eta = an.eta(theta, phi, Bvec, Xants, Xsource)
    omega = an.omega(theta, phi, Xants, Xsource)
    
    omega_cr = np.array([che.compute_Cerenkov(Xants[i,:], K, (groundAltitude-Xsource[2])/K[2], Xsource, 2e3)
                         for i in range(Xants.shape[0])])
    
    # limit for small theta
    if theta <70*np.pi/180: 
            omega_cr = np.minimum(omega_cr, 0.6*np.pi/180)
    
    adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
    adf *= 1. + asym*np.cos(eta) # 
    
    return eta, omega, omega_cr, l_ant, adf

def ADF_loss(params, Aants, Xants, Xsource, uncertainty=0.075):
    """Compute chi² for the ADF function.

    Parameters
    ----------
    params : sequence of float
        ``[theta, phi, delta_omega, amplitude]``.
    Aants : ndarray
        Measured peak amplitudes, shape (N,).
    Xants : ndarray
        Antenna positions, shape (N, 3).
    Xsource : ndarray
        Shower source position, shape (3,) (from SWF).
    uncertainty : float, optional
        Relative uncertainty on amplitudes (default: 7.5%).

    Returns
    -------
    float
        chi² value.
    """
    theta, phi, delta_omega, amplitude = params

    # Compute model
    _, _, _, _, amplitude_model = ADF_parameters(theta, phi, delta_omega, amplitude, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    
    chi2 = np.sum((Aants - amplitude_model)**2 / (uncertainty*Aants)**2)
    return chi2


def recons_ADF(theta_pwf, phi_pwf, Aants, Xants, Xsource):
    """
    Fit the ADF parameters to measured antenna peak amplitudes.
    
    This function performs a chi² minimization using the ADF_loss function to 
    reconstruct the best-fit shower parameters: direction (theta, phi), ADF width (delta_omega), 
    and amplitude.
    
    Inputs:
        theta_pwf : initial guess for shower zenith angle (rad)
        phi_pwf   : initial guess for shower azimuth angle (rad)
        Aants     : measured peak amplitudes at antennas (N,)
        Xants     : antenna positions (N,3)
        Xsource   : Xsource position (3,)
    
    Returns
    -------
        theta_adf     : reconstructed zenith angle (rad)
        phi_adf       : reconstructed azimuth angle (rad)
        delta_omega   : best-fit ADF shape parameter
        amplitude     : best-fit ADF amplitude
    """
    where = "recons_ADF"
    _checks.angles(where, theta_pwf=theta_pwf, phi_pwf=phi_pwf)
    Xants = _checks.antennas(Xants, where, min_ants=4)
    Aants = _checks.per_antenna(Aants, Xants, "Aants", where)
    # (1, 3), as compute_Xsource_cartesian_coords returns it, is accepted too
    Xsource = _validate.as_array(Xsource, "Xsource", where, finite=True)
    if Xsource.size != 3:
        raise ValueError(_validate.message(
            where, "'Xsource' must be three numbers (x, y, z), got shape %s" % (Xsource.shape,)))
    Xsource = Xsource.reshape(3)
    # The loss divides by the amplitudes: a zero made it infinite everywhere,
    # and the "fit" silently returned its starting point (#286)
    bad = np.nonzero(~(Aants > 0))[0]
    if bad.size:
        raise ValueError(_validate.message(
            where, "every amplitude in 'Aants' must be positive; antennas %s have %s (an "
            "antenna below one ADC count reads 0: leave it out of the fit)"
            % (bad.tolist(), Aants[bad].tolist())))
    # A source at an antenna or below the antennas has no meaningful
    # geometry; the minimizer ran for minutes, then returned the start (#286)
    if not Xsource[2] > np.max(Xants[:, 2]):
        raise ValueError(_validate.message(
            where, "'Xsource' must lie above the antennas (z %.1f m, highest antenna at %.1f m)"
            % (Xsource[2], np.max(Xants[:, 2]))))
    # Define bounds for each parameter
    bounds = [
        [theta_pwf - 2*np.pi/180, theta_pwf + 2*np.pi/180],
        [phi_pwf   - 1*np.pi/180, phi_pwf   + 1*np.pi/180],
        [1.25, 3.0],
        [1e6, 1e10]
    ]

    params_in = np.array(bounds).mean(axis=1)

    # Adjust initial amplitude guess using measured max amplitude and mean antenna distance
    l_ant = an.distance_source_antenna(Xants, Xsource)
    params_in[3] = Aants.max() * l_ant.mean()

    # Run minimization using iminuit
    result = minimize(
        ADF_loss,
        params_in,
        args=(Aants, Xants, Xsource),
        method="migrad",
        bounds=bounds
    )

    # A loss that is not finite means no fit took place (#286)
    if not np.isfinite(result.fun):
        raise RuntimeError(_validate.message(
            where, "the ADF fit failed: the loss is %s at the end, so the parameters "
            "returned would be the starting point" % result.fun))
    if not result.success:
        logger.warning("recons_ADF: the minimizer reports no convergence (loss %.3g); the "
                       "parameters may not be a minimum", result.fun)

    # Extract best-fit parameters from minimization result
    theta_adf, phi_adf, delta_omega, amplitude = result.x
    return theta_adf, phi_adf, delta_omega, amplitude

def ADF_fun(l_ant, amplitude, omega_cr, delta_omega):
    """Compute a simple model of the ADF for a shower on all omega values.

    Don't consider geomagnetic effect (very low) and average cherenkov angle along all eta values.

    Parameters
    ----------
    l_ant : float
        Mean distance from the antennas to the shower source (in meters). 
    amplitude : float
        Scaling factor of the ADF.
    omega_cr : float
        Mean Cherenkov opening angle (in radians).
    delta_omega : float
        Width parameter of the ADF.

    Returns
    -------
    omega : np.ndarray
        Array of angles (in radians) over which the ADF is evaluated.
    f_adf : np.ndarray
        Corresponding ADF amplitudes at each angle.
    """
    omega = np.linspace(0,3,200) * np.pi / 180
    f_adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
    return(omega,f_adf)