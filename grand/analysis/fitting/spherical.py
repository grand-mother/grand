import numpy as np
import grand.analysis.physics as phy
import grand.analysis.constants as cons
from scipy.optimize import differential_evolution
from numba import njit

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

@njit(**kwd)
def SWF_loss(theta, phi, r_xmax, t_s, Xants, tants, sigma = None, cr=cons.c_light):

    '''
    Defines Chi2 by summing model residuals over antennas  (i):
    loss = \sum_i ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

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
    sigma : float, optional
        Timing uncertainty (seconds). If provided, the chi2 is normalized by sigma^2.
    cr : float, optional
        Propagation speed of the signal (default: speed of light).

    Returns
    -------
    float
        Chi-square value (normalized if sigma is provided).
    '''
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    Xmax = -r_xmax * K + np.array([0.,0.,cons.groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape, Xants.shape)
        return None
    tmp = 0.
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = phy.ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
        tmp += res*res

    chi2 = tmp
    if sigma is not None:
        sigma = cr*sigma
    if sigma == None:
        return chi2
    return(chi2/ (sigma**2)) # chi^2( normalized by sigma^2 and degrees of freedom N-4)


def recons_swf(theta_pwf, phi_pwf, tants, Xants, sigma=None, maxiter=3000, seed=42, mu_xmax=None,
sigma_xmax=None, x0=None):
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
        Antenna positions, shape (N, 3).
    sigma : float, optional
        Timing uncertainty (seconds).
    maxiter : int, optional
        Maximum number of iterations for differential evolution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (theta_swf, phi_swf, r_xmax_swf, t_s_swf)
    """

    if (mu_xmax is None) != (sigma_xmax is None):
        raise ValueError("mu_xmax and sigma_xmax must be provided together")

    if mu_xmax is not None and sigma is None:
        raise ValueError("sigma must be provided when using the Xmax prior")
    
    if sigma_xmax is not None and sigma_xmax <= 0:
        raise ValueError("sigma_xmax must be positive")

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
    
    
    def objective(p):

        chi2 = SWF_loss(
            p[0], p[1], p[2], p[3],
            Xants, tants, sigma
        )

        if mu_xmax is not None:
            chi2 += Xmax_prior_loss(
                p[0], p[1], p[2],
                mu_xmax, sigma_xmax
            )

        return chi2


    # # Run the minimization
    # result = differential_evolution(
    #     lambda p: SWF_loss(p[0], p[1], p[2], p[3], Xants, tants, sigma),
    #     bounds=bounds,
    #     maxiter=maxiter,
    #     tol=1e-6,
    #     mutation=(0.5, 1),
    #     recombination=0.7,
    #     seed=seed,
    #     x0 = np.array(bounds, dtype=np.float64).mean(axis=1)
    #     # disp=True
    # )
    
    if x0 is None:
        x0 = np.array(bounds, dtype=np.float64).mean(axis=1)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
        x0 = x0
        # disp=True
    )
    
    # Extract best-fit parameters
    theta_swf, phi_swf, r_xmax_swf, t_s_swf = result.x
    
    return theta_swf, phi_swf, r_xmax_swf, t_s_swf 


def compute_Xsource_cartesian_coords(theta_swf, phi_swf, r_xmax, groundAltitude=cons.groundAltitude):
    """
    Compute the Cartesian coordinates of the emission point (Xsource)
    from spherical coordinates.

    Parameters
    ----------
    theta_swf : float
        Polar angle (radians).
    phi_swf : float
        Azimuth angle (radians).
    r_xmax : float
        Distance to the source (meters).
    groundAltitude : float, optional
        Ground altitude in the detector reference frame (meters).

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the source in GRAND detector frame, shape (1, 3).
    """
    st=np.sin(theta_swf); ct=np.cos(theta_swf); sp=np.sin(phi_swf); cp=np.cos(phi_swf) 
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
        Antenna positions in the detector reference frame, shape (N, 3).
    groundAltitude : float, optional
        Ground altitude in meters (default: cons.groundAltitude).
    cr : float, optional
        Propagation speed of the signal (default: speed of light).

    Returns
    -------
    np.ndarray
        Expected arrival times at each antenna, shape (N,).
    """
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([-st*cp, -st*sp, -ct])
    Xsource = -r_xsource * K + np.array([0., 0., groundAltitude])
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = phy.ZHSEffectiveRefractionIndex(Xsource, Xants[i, :])
        dX = Xants[i, :] - Xsource
        tants[i] = t_s + n_average / cr * np.linalg.norm(dX)
    return tants



# ----------------------------
# Add for regularisation
# ----------------------------

def compute_Xmax_grammage(theta, phi, r_xmax):
    """
    Compute atmospheric depth at the reconstructed SWF source.
    Returns Xmax in g/cm².
    """

    X0 = compute_Xsource_cartesian_coords(
        theta, phi, r_xmax
    )[0]

    Xground = np.array([
        0.,
        0.,
        cons.groundAltitude,
    ])

    # Direction from ground towards the source
    if r_xmax > 0:
        u = (X0 - Xground) / r_xmax
    else:
        X1 = compute_Xsource_cartesian_coords(
            theta, phi, 1.
        )[0]
        u = X1 - Xground

    return phy.SlantDepth(X0, u)

def Xmax_prior_loss(
    theta,
    phi,
    r_xmax,
    mu_xmax=750.,
    sigma_xmax=100.,
):
    """
    Gaussian prior on atmospheric Xmax.
    """

    # Prior intended for downward-going showers
    if theta > np.pi / 2:
        return 1e6

    Xmax = compute_Xmax_grammage(
        theta, phi, r_xmax
    )

    if not np.isfinite(Xmax):
        return 1e6

    return ((Xmax - mu_xmax) / sigma_xmax)**2




# Analytical prior on Xmax
def recons_swf_analytic(
    theta_pwf,
    phi_pwf,
    tants,
    Xants,
    sigma=5e-9,
    mu_xmax=None,
    sigma_xmax=None,
    maxiter=500,
    seed=42,
):
    """
    SWF reconstruction with analytical emission time.

    Optimize theta, phi and log10(r).
    The emission time is calculated analytically.

    Optional Gaussian prior on Xmax.
    """

    if sigma is None or sigma <= 0:
        raise ValueError("A positive timing uncertainty is required")

    if (mu_xmax is None) != (sigma_xmax is None):
        raise ValueError("mu_xmax and sigma_xmax must be provided together")

    if sigma_xmax is not None and sigma_xmax <= 0:
        raise ValueError("sigma_xmax must be positive")

    # Parameter bounds
    angle_window = np.deg2rad(5)

    theta_min = max(0.0, theta_pwf - angle_window)
    theta_max = min(np.pi / 2, theta_pwf + angle_window)

    if theta_min >= theta_max:
        raise ValueError("No downward-going direction in the PWF window")

    r_max = 2e6  # m

    bounds = [
        (theta_min, theta_max),
        (phi_pwf - angle_window, phi_pwf + angle_window),
        (np.log10(0.1), np.log10(r_max / 1e3)),
    ]

    # Calculate optimal emission time
    def compute_t_s(theta, phi, r):

        t_propagation = SWF_model(
            theta, phi, r, 0.0, Xants
        )

        t_s = np.mean(tants - t_propagation)

        t_s = np.clip(
            t_s,
            -r_max / cons.c_light,
            0.0
        )

        return t_s, t_propagation

    # Objective function
    def objective(params):

        theta, phi, log_r_km = params

        r = 10**log_r_km * 1e3

        t_s, t_propagation = compute_t_s(theta, phi, r)

        residuals = tants - t_s - t_propagation

        chi2 = np.sum((residuals / sigma)**2)

        if mu_xmax is not None:
            chi2 += Xmax_prior_loss(
                theta, phi, r,
                mu_xmax, sigma_xmax
            )

        return chi2

    # Initial parameters
    x0 = np.array([
        theta_pwf,
        phi_pwf,
        np.log10(30),
    ])

    lower, upper = np.asarray(bounds).T
    margin = 1e-8 * (upper - lower)
    x0 = np.clip(x0, lower + margin, upper - margin)

    # Minimization
    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=x0,
        popsize=10,
        maxiter=maxiter,
        tol=1e-6,
        seed=seed,
    )

    # Reconstructed parameters
    theta, phi, log_r_km = result.x

    r = 10**log_r_km * 1e3

    t_s, _ = compute_t_s(theta, phi, r)

    return theta, phi, r, t_s