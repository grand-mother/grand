import grand.analysis.coords.array_shower as co
import numpy as np
import grand.analysis.constants as cons

def eta(theta, phi, Bvec, Xants, Xsource):
    """
    Computes the angle eta (azimuth angle in the shower plane).

    theta  : shower zenith angle (rad)
    phi    : shower azimuth angle (rad)
    Bvec   : magnetic field vector (3,)
    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX_sp = co.to_shower_frame(theta, phi, Bvec, Xants, Xsource)
    if dX_sp.ndim == 1:
        return np.arctan2(dX_sp[1], dX_sp[0])
    else:
        return np.arctan2(dX_sp[:,1], dX_sp[:,0])

def distance_source_antenna(Xants,Xsource):
    """
    Computes the distance(s) between source and antenna(s).

    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX = Xants - Xsource 
    if dX.ndim == 1:
        return np.linalg.norm(dX)
    else:
        return np.linalg.norm(dX, axis=1)
    

def omega(theta, phi, Xants, Xsource):
    """
    Computes the angle omega between shower direction and antenna vector(s).

    theta  : shower zenith angle (rad)
    phi    : shower azimuth angle (rad)
    Xants  : antenna positions (3,) or (N,3)
    Xsource: Xsource position in GRAND detector frame (3,)
    """
    dX = Xants - Xsource 
    l_ant = distance_source_antenna(Xants,Xsource)
    K = co.shower_direction_vector(theta, phi)
    if dX.ndim == 1:
        cos_omega = np.dot(K, dX) / l_ant   
    else:
        cos_omega = np.einsum('i,ni->n', K, dX) / l_ant 

    return np.arccos(cos_omega)  

def sin_geomag_angle(theta, phi, B=cons.Bn):
    """
    Computes the sine of the geomagnetic angle (alpha) between the shower axis
    and the geomagnetic field.

    Parameters
    ----------
    theta : float or array-like
        Zenith angle(s) of the shower in radians.
    phi : float or array-like
        Azimuth angle(s) of the shower in radians.
    B : array-like, shape (3,), optional
        Geomagnetic field vector (default: cons.Bn).

    Returns
    -------
    sin_alpha : float or ndarray
        Sine of the geomagnetic angle
    """
    K = co.shower_direction_vector(theta, phi)
    sin_alpha = np.cross(K.T,B)
    sin_alpha = np.linalg.norm(sin_alpha)
    return sin_alpha
    


