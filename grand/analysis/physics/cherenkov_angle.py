import numpy as np
import grand.analysis.physics as atm
from numba import njit

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

@njit(**kwd)
def compute_Cerenkov(Xant, K, xsourceDist, Xsource, delta):

    """
    Compute Cherenkov angle by minimizing the time delay between light rays from shower points and the observer.

    Inputs:
    - Xant : np.array, shape (3,) -> antenna position in meters
    - K : np.array, shape (3,) -> shower direction (unit vector)
    - xsourceDist : float -> distance from Xsource to shower core along K
    - Xsource : np.array, shape (3,) -> position of Xsource 
    - delta : float -> distance along shower axis to points before/after Xmax

    Returns:
    - omega_cr : float -> Cherenkov angle in radians
    """
    # Compute coordinates of point before Xmax
    Xb = Xsource - delta*K
    # Compute coordinates of point after Xmax
    Xa = Xsource + delta*K

    # Compute core position at ground altitude
    theta = np.arccos(-K[2])
    phi = np.arctan2(-K[1], -K[0])
    z_core = Xant[2]
    y_core = np.sin(theta)*np.sin(phi)*(z_core-Xsource[2])/(-np.cos(theta)) + Xsource[1]
    x_core = np.sin(theta)*np.cos(phi)*(z_core-Xsource[2])/(-np.cos(theta)) + Xsource[0]
    Xcore = np.empty(3, dtype=np.float64)
    Xcore[0] = x_core
    Xcore[1] = y_core
    Xcore[2] = z_core

    dXcore = Xant - Xcore

    # Direction vector to observer's position from shower core
    # This is a bit dangerous for antennas numerically close to shower core... 
    U = dXcore / np.maximum(np.linalg.norm(dXcore), 1e-10)
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.arccos(np.dot(K,U))
    alpha = np.pi-alpha


    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax)) 
    omega_cr_guess = np.arccos(1./atm.RefractionIndexAtPosition(Xsource))
    # print("###############")
    # omega_cr = fsolve(compute_delay,[omega_cr_guess])
    omega_cr = newton(compute_delay, omega_cr_guess, args=(Xsource, Xa, Xb, Xant,U,K,alpha,delta, xsourceDist),verbose=False)
    ### DEBUG ###
    # omega_cr = omega_cr_guess
    # Reject non-converged (NaN) or unphysical roots, e.g. when theta+omega > 90 deg

    return(omega_cr)

@njit(**kwd)
def compute_delay(omega,Xmax, Xa, Xb,Xant,U,K,alpha,delta,xmaxDist):
    """
    Compute residual time delay for a given Cherenkov angle guess.

    Inputs:
    - omega : float -> guessed Cherenkov angle (radians)
    - Xmax : np.array, shape (3,) -> shower maximum position
    - Xa, Xb : np.array, shape (3,) -> points before/after Xmax along shower axis
    - Xant : np.array, shape (3,) -> antenna position
    - U : np.array, shape (3,) -> unit vector from shower core to antenna
    - K : np.array, shape (3,) -> shower direction (unit vector)
    - alpha : float -> angle between shower direction and vector to antenna
    - delta : float -> distance along shower axis from Xmax
    - xmaxDist : float -> distance from Xmax to shower core along K

    Returns:
    - res : float -> residual (should be zero for correct Cherenkov angle)
    """

    X = compute_observer_position(omega,Xmax,Xant,U,K,xmaxDist,alpha)
    # print('omega = ',omega,'X_obs = ',X)
    n2 = atm.ZHSEffectiveRefractionIndex(Xa,X)
    # print('n0 = ',n0)
    n1 = atm.ZHSEffectiveRefractionIndex(Xb, X)
    # print('n1 = ',n1)
    res = minor_equation(omega,n2,n1,alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)

@njit(**kwd)
def minor_equation(omega, n2, n1, alpha, delta, xmaxDist):

    '''
    Compute time delay (in m)
    Compute [c*delta(t)]^2    
    '''
    sa = np.sin(alpha)
    saw = np.sin(alpha+omega)
    saw = saw if np.abs(saw) > 1e-10 else 1e-10 # Avoid division by zero when observer ray is parallel to U
    com = np.cos(omega)
    l0 = xmaxDist*sa/saw
    l1 = np.sqrt(l0**2+delta**2+2*delta*l0*com)
    l2 = np.sqrt(l0**2+delta**2-2*delta*l0*com)
    # Eq. 3.38 p125.
    res = (n2*l2+2*delta)**2-(n1*l1)**2
    return(res)

@njit(**kwd)
def compute_observer_position(omega,Xmax,Xant,U,K,xmaxDist,alpha):
    """
    Compute observer (antenna) position given Cherenkov angle.

    Inputs:
    - omega : float -> Cherenkov angle
    - Xmax : np.array, shape (3,) -> shower maximum position
    - Xant : np.array, shape (3,) -> antenna position
    - U : np.array, shape (3,) -> unit vector from shower core to antenna
    - K : np.array, shape (3,) -> shower direction
    - xmaxDist : float -> distance from Xmax to shower core
    - alpha : float -> angle between shower direction and vector to antenna

    Returns:
    - X : np.array, shape (3,) -> computed observer position
    """

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    # this assumed coincidence was computed at antenna altitude)
    #t = (Xant[2] - Xmax[2])/Dir_obs[2]
    # This assumes coincidence is computed at fixed alpha, i.e. along U, starting from Xcore
    saw = np.sin(alpha+omega)
    saw = saw if np.abs(saw) > 1e-10 else 1e-10 # Avoid division by zero when observer ray is parallel to U
    t = np.sin(alpha)/saw * xmaxDist
    X = Xmax + t*Dir_obs
    return (X)

@njit(**kwd)
def rotation(angle,axis):
    """
    Compute 3x3 rotation matrix around axis using Rodrigues formula.

    Inputs:
    - angle : float -> rotation angle in radians
    - axis : np.array, shape (3,) -> unit rotation axis

    Returns:
    - mat : np.array, shape (3,3) -> rotation matrix
    """
    ca = np.cos(angle)
    sa = np.sin(angle)

    cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    mat = np.eye(3) + sa*cross + (1.0-ca)*np.dot(cross,cross)
    return (mat)

@njit(**kwd)
def der(func,x,args=[], eps=1e-7):
    '''
    Forward estimate of derivative
    '''
    return ((func(x+eps,*args)-func(x,*args))/eps)

@njit(**kwd)
def newton(func,x0,tol=1e-7,nstep_max = 100, args = [], verbose=False):
    """
    Newton-Raphson zero-finding method with numerical derivative.

    Inputs:
    - func : callable -> function for which zero is sought
    - x0 : float -> initial guess
    - tol : float -> relative tolerance
    - nstep_max : int -> maximum iterations
    - args : list -> extra arguments to func
    - verbose : bool -> print iteration info

    Returns:
    - x : float -> estimated zero of func (NaN if not converged)
    """
    rel_error = np.inf
    xold = x0
    nstep = 0
    while ((rel_error > tol) and (nstep<nstep_max)):

        # der() evaluated once per iteration (it was computed twice: in the test and in the value)
        d = der(func,xold,args=args)
        derivative = d if np.abs(d) > 1e-12 else 1e-12  # Avoid division by zero

        # derivative = der(func,xold,args=args) if np.abs(der(func,xold,args=args)) > 1e-12 else 1e-12  # Avoid division by zero
        x = xold - func(xold,*args)/derivative
        nstep += 1
        if verbose==True:
            print ("x at iteration",nstep, 'is ',x)
        if np.abs(xold) > 1e-12:
            rel_error = np.abs((x-xold)/xold)
        else:
            rel_error = np.abs(x-xold)
        xold = x
    return(x)
