"""Spherical-wave fit of the arrival direction and source distance."""

import numpy as np
import grand.analysis.physics as phy
import grand.analysis.constants as cons
from scipy.optimize import differential_evolution
from grand.analysis import _checks
from grand.basis import validate as _validate


def SWF_loss(theta, phi, r_xmax, t_s, Xants, tants, sigma = None, cr=cons.c_light):
    r"""Define Chi2 by summing model residuals over antennas (i).

    The loss is::

        loss = sum_i ( cr (tants[i] - t_s) - n_i |Xants[i] - X_s| )**2
        X_s  = r_xmax * (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
               + (0, 0, groundAltitude)

    where `Xants` are the antenna positions (shape (N, 3)), `tants` the
    trigger times (shape (N,)) and n_i the mean refractive index along the
    path.  The source lies at -r_xmax * K, on the side the shower
    *comes from* (K is the propagation direction; #216).

    Parameters
    ----------
    theta : float
        Polar angle of the shower direction (radians).
    phi : float
        Azimuth angle of the shower direction (radians).
    r_xmax : float
        Distance from the ground to the emission point (meters).
    t_s : float
        Emission time of the source (seconds).
    Xants : np.ndarray
        Antenna positions in the detector reference frame, shape (N, 3).
    tants : np.ndarray
        Measured signal times at antennas (seconds), shape (N,).
    sigma : float, ndarray of shape (N,) or (N, N), optional
        Timing uncertainty (seconds): one for all antennas, one per antenna,
        or their covariance matrix (seconds squared).  If provided, each
        residual is divided by its uncertainty (#289).
    cr : float, optional
        Propagation speed of the signal (default: speed of light).

    Returns
    -------
    float
        Chi-square value (normalized if sigma is provided).
    """
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,cons.groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Make sure Xants and tants are compatible (it used to print and return None)
    if Xants.ndim != 2 or Xants.shape[1] != 3 or Xants.shape[0] != nants:
        raise ValueError(_validate.message(
            "SWF_loss", "'Xants' must have shape (N, 3) with one row per arrival time (%d), "
            "got %s" % (nants, Xants.shape)))
    res = np.empty(nants)
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = phy.ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res[i] = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)

    if sigma is None:
        return float(res @ res)
    # Residuals are in meters: the timing uncertainty becomes cr*sigma.  A
    # vector of per-antenna uncertainties gave an array, not a chi2 (#289)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2:
        return float(res @ np.linalg.solve(cr**2 * sigma, res))
    return float(np.sum((res / (cr * sigma))**2))


def recons_swf(theta_pwf, phi_pwf, tants, Xants, sigma=None, maxiter=1000, seed=42):
    """
    Perform a SWF reconstruction using differential evolution minimization.

    The minimization is performed around PWF direction to improve convergence.

    Parameters
    ----------
    theta_pwf : float
        Initial zenith angle from Plane Wave Fit (radians).
    phi_pwf : float
        Initial azimuth angle from Plane Wave Fit (radians).
    tants : np.ndarray
        Measured antenna times (seconds).
    Xants : np.ndarray
        Antenna positions, shape (N, 3), in meters: x North, y West, z above
        sea level.
    sigma : float, ndarray of shape (N,) or (N, N), optional
        Timing uncertainty (seconds), as for :func:`SWF_loss`.  A single
        value scales the chi2 without moving its minimum; per-antenna values
        weight the antennas (#289).
    maxiter : int, optional
        Maximum number of iterations for differential evolution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(theta_swf, phi_swf, r_xmax_swf, t_s_swf)``: the direction the
        shower comes from, in radians; the distance from the source to
        ``(0, 0, groundAltitude)``, in meters; and the emission time, in
        seconds (#261).
    """
    where = "recons_swf"
    _checks.angles(where, theta_pwf=theta_pwf, phi_pwf=phi_pwf)
    Xants = _checks.antennas(Xants, where, min_ants=4)
    tants = _checks.per_antenna(tants, Xants, "tants", where)
    sigma = _checks.sigma(sigma, where)
    # Parameter bounds for the differential evolution
    bounds = [[theta_pwf-5*np.pi/180,theta_pwf+5*np.pi/180],
                [phi_pwf-5*np.pi/180,phi_pwf+5*np.pi/180], 
                [0, 2000000],
                [-2000000/cons.c_light, 0]]
    #            [-15.6e3 - 12.3e3/np.cos(np.pi - theta_pwf),-6.1e3 - 15.4e3/np.cos(np.pi - theta_pwf)],
    #            [(6.1e3 + 15.4e3/np.cos(np.pi - theta_pwf)) / cons.c_light, 0]] 
    
    #bounds = [[np.deg2rad(0),np.deg2rad(180)],
    #        [np.deg2rad(0),np.deg2rad(360)], 
    #        [0, 2000000],
    #        [-2000000/cons.c_light, 0]]
    
    
    # Run the minimization.  sigma was documented but ignored: both
    # branches of an if passed the loss without it (#289)
    result = differential_evolution(
        lambda p: SWF_loss(p[0], p[1], p[2], p[3], Xants, tants, sigma=sigma),
        bounds=bounds,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed
    )

    # Extract best-fit parameters
    theta_swf, phi_swf, r_xmax_swf, t_s_swf = result.x
    
    return theta_swf, phi_swf, r_xmax_swf, t_s_swf 


def compute_Xsource_cartesian_coords(theta_swf, phi_swf, r_xmax, groundAltitude=cons.groundAltitude):
    """Compute the Cartesian coordinates of the emission point (Xsource).

    From spherical coordinates.

    Parameters
    ----------
    theta_swf : float
        Polar angle (radians).
    phi_swf : float
        Azimuth angle (radians).
    r_xmax : float
        Distance to the source (meters).
    groundAltitude : float, optional
        Height above sea level of the frame's origin, where the source
        distance is measured from (meters).  The default, 1231 m, is the GP13
        site; for simulation files pass the ground altitude
        :func:`grand.analysis.geom.antenna_positions_from_run` returns (#252).

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the source in GRAND detector frame, shape (1, 3).
    """
    st = np.sin(theta_swf)
    ct = np.cos(theta_swf)
    sp = np.sin(phi_swf)
    cp = np.cos(phi_swf)
    K = [-st*cp,-st*sp,-ct]
    Xsource = np.column_stack((-r_xmax*K[0], -r_xmax*K[1], groundAltitude-r_xmax*K[2]))
    return Xsource

def SWF_model(theta, phi, r_xsource, t_s, Xants, groundAltitude=cons.groundAltitude, cr=cons.c_light):
    """
    Compute the expected arrival times at each antenna based on the Spherical Wave Front (SWF) model.

    The emission point (Xsource) is located at a distance r_xsource along the direction opposite
    to the shower propagation, referenced from the ground altitude. Arrival times are computed
    assuming straight-line propagation with an average refraction index.

    Parameters
    ----------
    theta : float
        Polar angle of the shower direction (radians).
    phi : float
        Azimuth angle of the shower direction (radians).
    r_xsource : float
        Distance from the shower maximum (emission point) to the reference point at the ground (meters).
    t_s : float
        Emission time of the source (seconds).
    Xants : np.ndarray
        Antenna positions, shape (N, 3): x North, y West, z the height above
        sea level, in meters.
    groundAltitude : float, optional
        Height above sea level of the frame's origin, where the source
        distance is measured from (meters).  The default, 1231 m, is the GP13
        site; for simulation files pass the ground altitude
        :func:`grand.analysis.geom.antenna_positions_from_run` returns (#252).
    cr : float, optional
        Propagation speed of the signal (default: speed of light).

    Returns
    -------
    np.ndarray
        Expected arrival times at each antenna, shape (N,).
    """
    nants = Xants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([-st*cp, -st*sp, -ct])
    Xsource = -r_xsource * K + np.array([0., 0., groundAltitude])
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = phy.ZHSEffectiveRefractionIndex(Xsource, Xants[i, :])
        dX = Xants[i, :] - Xsource
        tants[i] = t_s + n_average / cr * np.linalg.norm(dX)
    return tants

