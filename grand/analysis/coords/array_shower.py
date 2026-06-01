import numpy as np

def shower_direction_vector(theta, phi):
    """
    Returns the shower direction vector K (orientation of the shower front).

    theta : zenith angle in radians
    phi   : azimuth angle in radians

    Returns:
        K : numpy array of shape (3,)
    """

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    K = np.array([
        -st * cp,
        -st * sp,
        -ct
    ])

    return K

def transformation_matrix(theta, phi, Bvec):
    """
    Builds the rotation matrix to the shower frame.

    theta : zenith angle (rad)
    phi   : azimuth angle (rad)
    Bvec  : global magnetic field vector (3,)
    """

    K = shower_direction_vector(theta, phi)

    KxB = np.cross(K, Bvec)
    KxB /= np.linalg.norm(KxB)

    KxKxB = np.cross(K, KxB)
    KxKxB /= np.linalg.norm(KxKxB)

    M = np.vstack((KxB, KxKxB, K))   # 3x3

    return M

def to_shower_frame(theta, phi, Bvec, Xant, Xsource):
    """
    Transforms global coordinates -> shower frame.

    Xant     : (3,) or (N,3) antenna positions
    Xsource  : (3,) core/Xmax position
    Returns:
        X_shower : coordinates in the shower frame
    """

    M = transformation_matrix(theta, phi, Bvec)
    dX = Xant - Xsource              # N x 3
    # projection onto shower frame
    X_shower = dX @ M.T              # N x 3 (=np.dot(mat,dX))
    return X_shower
