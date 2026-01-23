import numpy as np
import grand.analysis.physics as phy
import grand.analysis.constants as cons
from scipy.optimize import differential_evolution


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
    return(chi2/(sigma**2))


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

    # Parameter bounds for the differential evolution
    bounds = [[theta_pwf-5*np.pi/180,theta_pwf+5*np.pi/180],
                [phi_pwf-5*np.pi/180,phi_pwf+5*np.pi/180], 
                [-15.6e3 - 12.3e3/np.cos(np.pi - theta_pwf),-6.1e3 - 15.4e3/np.cos(np.pi - theta_pwf)],
                [(6.1e3 + 15.4e3/np.cos(np.pi - theta_pwf)) / cons.c_light, 0]] 
    

    # Run the minimization
    if sigma is None: 
        result = differential_evolution(
        lambda p: SWF_loss(p[0], p[1], p[2], p[3], Xants, tants),
        bounds=bounds,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed
    )

    else:
        result = differential_evolution(
            lambda p: SWF_loss(p[0], p[1], p[2], p[3], Xants, tants, sigma),
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

