import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import grand.analysis.constants as cons


def compute_core(k, Xsource, groundAltitude=cons.groundAltitude):
    """
    Compute the shower core position as the intersection of the shower axis
    with the ground plane.

    Warning
    -------
    This function assumes a flat ground surface and does not take local
    topography into account.
    TODO: extend the implementation to include realistic ground elevation models.

    Parameters
    ----------
    k : ndarray, shape (3,)
        Shower direction unit vector.
    Xsource : ndarray, shape (3,)
        Emission point coordinates [m].
    groundAltitude : float
        Reference altitude defining the ground plane (meters).

    Returns
    -------
    xc : ndarray, shape (3,)
        Core position at ground [m].
    """
    k = np.asarray(k).reshape(3,)    
    Xsource = np.asarray(Xsource).reshape(3,) 
    u = np.linspace(0, np.linalg.norm(Xsource*1.5), 5001)  # Distance from x0 (meters)
    traj = np.zeros((3,len(u)))
    traj[0,:] = k[0] * u + Xsource[0]
    traj[1,:] = k[1] * u + Xsource[1]
    traj[2,:] = k[2] * u + Xsource[2]
    
    u_core = (groundAltitude - Xsource[2]) / k[2]
    xc = Xsource + k * u_core
    return(xc)

def generate_cone_surface_vectors(k, omega, n=10):
    """
    Generate vectors uniformly distributed on a cone surface.
    This cone represents the Cherenkov emission cone. 
    The cone axis is aligned with the shower direction, and the opening angle corresponds to the Cherenkov angle.
    The generated vectors sample the surface of this cone and are used to compute the projected Cherenkov footprint on the ground.

    Parameters
    ----------
    k : ndarray, shape (3,)
         Unit vector defining the cone axis (shower propagation direction).
    omega : float
       Cone opening angle in radians (Cherenkov angle).
    n : int, optional
        Number of vectors generated uniformly around the cone (default: 10).

    Returns
    -------
    vectors : ndarray, shape (n, 3)
        Unit vectors lying on the surface of the Cherenkov cone, aligned
    with the shower axis.
    """

    # Generate n uniform angles around the axis (azimuthal angle)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Spherical to Cartesian conversion with fixed polar angle omega
    x = np.sin(omega) * np.cos(theta)
    y = np.sin(omega) * np.sin(theta)
    z = np.full_like(x, np.cos(omega))  # fixed z for all points (cos(omega))

    # These are vectors on a cone aligned with the z-axis
    vectors = np.vstack((x, y, z)).T  # shape (n, 3)

    # We now compute the rotation matrix that aligns z-axis to vector k
    z_axis = np.array([0, 0, 1])
    if np.allclose(k, z_axis):
        rot_matrix = np.eye(3)  # No rotation needed
    elif np.allclose(k, -z_axis):
        # Special case: 180° rotation around any perpendicular axis
        rot_matrix = np.array([[-1, 0, 0],
                               [0, -1, 0],
                               [0,  0, 1]])
    else:
        # Use Rodrigues' rotation formula to get rotation matrix
        v = np.cross(z_axis, k)
        c = np.dot(z_axis, k)
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rot_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    # Rotate all vectors from z-aligned cone to k-aligned cone
    rotated_vectors = vectors @ rot_matrix.T
    return rotated_vectors