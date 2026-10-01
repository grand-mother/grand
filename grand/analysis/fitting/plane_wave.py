"""Plane-wave fit of the arrival direction from antenna times."""


import grand.analysis.constants as cons
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from grand.analysis import _checks
from grand.basis import validate as _validate

def PWF_semianalytical(Xants, tants, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=None):
    """Solve the minimization problem using a semi-analytical approach.

    (see section 2.1.2)

    Parameters
    ----------
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    verbose (bool): Verbose output, default is False.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns
    -------
    ndarray: Theta and phi angles in radians.
    """
    where = "PWF_semianalytical"
    Xants = _checks.antennas(Xants, where, min_ants=3)
    tants = _checks.per_antenna(tants, Xants, "tants", where)
    sigma = _checks.sigma(sigma, where)
    # Times in ns where seconds are expected gave another direction (#266)
    if np.ptp(tants) > 1e-3:
        import warnings
        warnings.warn(_validate.message(where, "the peak times span %g s, more than a shower front "
                                        "takes to cross any array; are they in ns rather than s?"
                                        % np.ptp(tants)), _validate.GRANDlibWarning, stacklevel=2)

    PXT = Xants - mean(Xants, sigma)[None, :]
    # PXT = PXT - mean(PXT, sigma)[None, :]   #twice for numerical stability
    t_center = tants - mean(tants, sigma)
    # t_center = t_center - mean(t_center, sigma) #twice for numerical stability
    A = np.dot(PXT.T, PXT)
    b = np.dot(PXT.T, t_center) * c / n
    d, W = np.linalg.eigh(A)
    # Antennas on one line leave the direction undetermined about that line:
    # it returned [nan nan] with only a RuntimeWarning (#288)
    if d[1] <= 1e-12 * d[2]:
        raise ValueError(_validate.message(
            where, "the antennas lie on one line; a plane-wave direction needs them spread in two dimensions"))
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    # Equal times: the front is parallel to the array, and the shower comes
    # along its normal.  The solver divided by |beta| = 0 and failed, for
    # every vertical shower over a flat array (#288)
    if nbeta == 0 or np.ptp(tants) == 0:
        k_opt = W[:, 0] if W[2, 0] < 0 else -W[:, 0]
        theta_opt = np.arccos(np.clip(-k_opt[2], -1.0, 1.0))
        phi_opt = np.arctan2(-k_opt[1], -k_opt[0]) % (2 * np.pi)
        return np.array([theta_opt, phi_opt])

    if (np.abs(beta[0] / nbeta) < 1e-14):
        if (verbose):
            print("Degenerate case")
        mu = -d[0]
        c_ = np.zeros(3)
        c_[1] = beta[1] / (d[1] + mu)
        c_[2] = beta[2] / (d[2] + mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.])))
        c_[0] = -si * np.sqrt(1 - c_[1]**2 - c_[2]**2)
        k_opt = np.dot(W, c_)

    else:
        def nc(mu):
            c_ = beta / (d + mu)
            return ((c_**2).sum() - 1.)
        mu_min = -d[0] + beta[0]
        mu_max = -d[0] + np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        c_ = beta / (d + mu_opt)
        k_opt = np.dot(W, c_)

    if k_opt[2] > 1e-2:
        k_opt = k_opt - 2 * (k_opt @ W[:, 0]) * W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2 * np.pi
    return np.array([theta_opt, phi_opt])

def mean(X:np.ndarray, sigma=None):
    """Return the mean of ``X`` along its first axis, weighted by ``sigma``.

    Parameters
    ----------
    X : ndarray
        Values, one row per antenna.
    sigma : ndarray, optional
        Uncertainties: a vector weights each row by ``1/sigma``, a covariance
        matrix by the column sums of its inverse.  Unweighted if omitted.

    Returns
    -------
    ndarray
        The (weighted) mean.
    """
    if type(sigma) is np.ndarray and sigma.ndim==1:
        return ( 1/(1/sigma).sum() ) * ( (1/sigma) @ X )
    elif type(sigma) is np.ndarray and sigma.ndim==2:
        Q_1 = _inv_cho(sigma)
        return ( 1/Q_1.sum() ) * ( Q_1.sum(axis=0) @ X )
    else:
        return X.mean(axis=0)

def _inv_cho(A):
    c, low = cho_factor(A)
    A_inv = cho_solve((c, low), np.eye(A.shape[0]))
    return A_inv

def PWF_loss(params, Xants, tants, verbose=False, c=cons.c_light,  n=cons.n_atm, sigma=None):
    """Define Chi2 by summing model residuals over individual antennas.

    After maximizing likelihood over reference time.
    """
    where = "PWF_loss"
    Xants = _checks.antennas(Xants, where)
    tants = _checks.per_antenna(tants, Xants, "tants", where)
    if sigma is None:
        raise TypeError(_validate.message(
            where, "'sigma', the timing uncertainty in seconds, is required to normalise the chi2"))
    sigma = _checks.sigma(sigma, where)
    residuals = PWF_residuals(params, Xants, tants, verbose=verbose, c=c, n=n)
    chi2 = (residuals**2).sum()
    sigma = c*sigma #express in m
    #return(chi2/(sigma**2*(nants-2)))
    return(chi2/(sigma**2))

def PWF_residuals(params, Xants, tants, verbose=False, c=cons.c_light,  n=cons.n_atm):
    """Compute timing residuals for each antenna using the plane wave model.

    Note that this is defined at up to an additive constant, that when minimizing
    the loss over it, amounts to centering the residuals.
    """
    where = "PWF_residuals"
    Xants = _checks.antennas(Xants, where)
    tants = _checks.per_antenna(tants, Xants, "tants", where)

    times = PWF_model(params, Xants, c, n)
    res = (c/n) * (tants - times)
    res -= res.mean()  # Mean is projected out when maximizing likelihood over reference time t0
    return (res)

def PWF_model(params, Xants, c=cons.c_light,  n=cons.n_atm, groundAltitude=cons.groundAltitude):
    """Generate plane wavefront timings.

    Parameters
    ----------
    params : sequence
        ``(theta, phi)`` in radians.
    Xants : ndarray, shape (N, 3)
        Antenna positions: x North, y West, z the height above sea level (m).
    c, n : float, optional
        Speed of light and refractive index.
    groundAltitude : float, optional
        Height above sea level of the frame's origin, where the source
        distance is measured from (meters).  The default, 1231 m, is the GP13
        site; for simulation files pass the ground altitude
        :func:`grand.analysis.geom.antenna_positions_from_run` returns (#252).
        For the plane wave it only shifts all times by one constant.

    Returns
    -------
    ndarray, shape (N,)
        Arrival times in seconds, relative to the origin's.
    """
    theta, phi = params
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    dX = Xants - np.array([0.,0., groundAltitude])
    tants = np.dot(dX,K) / (c / n)
 
    return (tants)