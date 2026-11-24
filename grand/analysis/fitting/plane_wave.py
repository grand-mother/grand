
import grand.analysis.constants as cons
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import fsolve, brentq

def PWF_semianalytical(Xants, tants, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=None):
    """
    Solve the minimization problem using a semi-analytical approach.
    (see section 2.1.2)

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    verbose (bool): Verbose output, default is False.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Theta and phi angles in radians.
    """
    nants = tants.shape[0]

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    PXT = Xants - mean(Xants, sigma)[None, :]
    # PXT = PXT - mean(PXT, sigma)[None, :]   #twice for numerical stability
    t_center = tants - mean(tants, sigma)
    # t_center = t_center - mean(t_center, sigma) #twice for numerical stability
    A = np.dot(PXT.T, PXT)
    b = np.dot(PXT.T, t_center) * c / n
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

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
    '''
    Defines Chi2 by summing model residuals over individual antennas, 
    after maximizing likelihood over reference time.
    '''
    nants = tants.shape[0]
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Make sure tants and Xants are compatible
    residuals = PWF_residuals(params, Xants, tants, verbose=verbose, c=c, n=n)
    chi2 = (residuals**2).sum()
    sigma = c*sigma #express in m
    #return(chi2/(sigma**2*(nants-2)))
    return(chi2/(sigma**2))

def PWF_residuals(params, Xants, tants, verbose=False, c=cons.c_light,  n=cons.n_atm):

    '''
    Computes timing residuals for each antenna using plane wave model
    Note that this is defined at up to an additive constant, that when minimizing
    the loss over it, amounts to centering the residuals.
    '''
    nants = tants.shape[0]
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    times = PWF_model(params, Xants, c, n)
    res = (c/n) * (tants - times)
    res -= res.mean()  # Mean is projected out when maximizing likelihood over reference time t0
    return (res)

def PWF_model(params, Xants, c=cons.c_light,  n=cons.n_atm, groundAltitude=cons.groundAltitude):
    '''
    Generates plane wavefront timings
    '''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    K = np.array([-st*cp,-st*sp,-ct])
    dX = Xants - np.array([0.,0., groundAltitude])
    tants = np.dot(dX,K) / (c / n)
 
    return (tants)