# -*- coding: utf-8 -*-
r"""Input checks shared by the reconstruction functions in ``grand.analysis``."""

import numpy as np

from grand.basis import validate as _validate


def antennas(Xants, where, min_ants=1, name="Xants"):
    r"""Checks antenna positions: finite, shape (N, 3), at least `min_ants` rows.

    Parameters
    ----------
    Xants : array-like
        Antenna positions, in metres.
    where : str
        The calling function, for messages.
    min_ants : int, optional
        The fewest antennas the calculation can use.
    name : str, optional
        The argument's name, for messages.

    Returns
    -------
    numpy.ndarray
        The positions as floats, shape (N, 3).
    """
    Xants = _validate.as_array(Xants, name, where, shape=(None, 3), finite=True)
    if len(Xants) < min_ants:
        raise ValueError(_validate.message(
            where, "needs at least %d antennas, got %d" % (min_ants, len(Xants))))
    return Xants


def per_antenna(values, Xants, name, where, finite=True):
    r"""Checks one value per antenna: a 1-D array as long as `Xants`.

    Parameters
    ----------
    values : array-like
        For example the arrival times or the peak amplitudes.
    Xants : numpy.ndarray
        The checked antenna positions.
    name : str
        The argument's name, for messages.
    where : str
        The calling function, for messages.
    finite : bool, optional
        Refuse NaN and infinity.

    Returns
    -------
    numpy.ndarray
    """
    values = _validate.as_array(values, name, where, ndim=1, finite=finite)
    if len(values) != len(Xants):
        raise ValueError(_validate.message(
            where, "'%s' must have one value per antenna (%d), got %d"
            % (name, len(Xants), len(values))))
    return values


def angles(where, **values):
    r"""Checks angles in radians are finite real numbers (scalars or arrays).

    Parameters
    ----------
    where : str
        The calling function, for messages.
    **values : float or array-like
        The angles, by argument name.
    """
    for name, value in values.items():
        if np.ndim(value) == 0:
            _validate.as_real(value, name, where)
        else:
            _validate.as_array(value, name, where, finite=True)
        # Degrees given where radians are expected went through as radians (#266)
        _validate.plausible(value, name, where, "angle_rad")


def sigma(value, where, name="sigma"):
    r"""Checks a timing or amplitude uncertainty: ``None``, positive, or a covariance matrix.

    A number or a vector must be positive.  A square matrix is a covariance
    matrix: its off-diagonal terms may be zero or negative, so only its
    diagonal (the variances) must be positive.

    Returns
    -------
    None, float or numpy.ndarray
    """
    if value is None:
        return None
    if np.ndim(value) == 0:
        value = _validate.as_real(value, name, where)
    else:
        value = _validate.as_array(value, name, where, finite=True)
    if np.ndim(value) == 2:
        if value.shape[0] != value.shape[1]:
            raise ValueError(_validate.message(
                where, "'%s' given as a matrix must be a square covariance matrix, got shape %s"
                % (name, value.shape)))
        _validate.positive(np.diag(value), "the diagonal of '%s' (the variances)" % name, where)
    else:
        _validate.positive(value, name, where)
    return value
