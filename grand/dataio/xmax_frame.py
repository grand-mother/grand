# -*- coding: utf-8 -*-
r"""Which vertical frame a file's ``xmax_pos_shc`` is in, read from the file.

``xmax_pos_shc`` is Xmax in shower-core coordinates, so its ``z`` should be a
height above the ground.  Files written before the ZHAireS converter
subtracted the ground altitude -- including every sample committed under
``sim2root/Common/`` -- carry the height above sea level instead
(grand-mother/grand#160).  Nothing in a file says which it is.

The geometry does.  Xmax lies on the shower axis, so the vector from the core
to Xmax points where the shower comes from, which the same tree stores as
``zenith`` and ``azimuth``.  In the right frame the two agree; with the ground
altitude still in ``z`` they do not.  On all twelve committed ZHAireS events
the ground-relative vector matches the stored direction to 0.001 degrees and
the raw one misses by 0.49 to 7.0 degrees.

Decided 2026-09-24: readers detect the frame, rather than the samples being
regenerated.  Data from DC2 was written under the old convention too.
"""

import logging

import numpy as np

from grand.basis import validate as _validate

logger = logging.getLogger(__name__)

#: How closely the core-to-Xmax vector must follow the stored direction, in
#: degrees.  Measured agreement is 0.001; the smallest miss is 0.49.
TOLERANCE_DEG = 0.05

GROUND = "ground"
SEA_LEVEL = "sea-level"
UNDETERMINED = "undetermined"


def _angles(zenith, azimuth, where):
    r"""Checks a shower direction in degrees; returns it as two floats."""
    zenith = _validate.as_real(zenith, "zenith", where)
    azimuth = _validate.as_real(azimuth, "azimuth", where)
    _validate.in_range(zenith, "zenith", where, 0, 180, "degrees")
    return zenith, azimuth


def _vector3(value, name, where):
    r"""Checks a position is three numbers, x, y and z (NaN allowed: it means unknown)."""
    array = _validate.as_array(value, name, where)
    if array.size != 3:
        raise ValueError(_validate.message(
            where, "'%s' must be three numbers (x, y, z), got shape %s" % (name, array.shape)))
    return array.reshape(3)


def arrival_direction(zenith, azimuth):
    r"""Unit vector pointing to where the shower comes from.

    Parameters
    ----------
    zenith, azimuth : float
        The shower's direction in degrees, "comes from", as stored in
        ``tshower`` (see ``tests/geo/test_angle_convention.py``).

    Returns
    -------
    numpy.ndarray
        Shape (3,), x North, y West, z Up.  Upwards for a downgoing shower.
    """
    zenith, azimuth = _angles(zenith, azimuth, "arrival_direction")
    theta, phi = np.deg2rad(zenith), np.deg2rad(azimuth)
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def propagation_direction(zenith, azimuth):
    r"""Unit vector along which the shower travels: ``-arrival_direction``.

    This is what ``tshower.direction`` stores, and the same vector the
    ZHAireS converter writes as ``primary_inj_dir_shc``.

    Parameters
    ----------
    zenith, azimuth : float
        The shower's direction in degrees, "comes from".

    Returns
    -------
    numpy.ndarray
        Shape (3,), x North, y West, z Up.  Downwards for a downgoing shower.
    """
    return -arrival_direction(zenith, azimuth)


def _angle_to_direction(vector, zenith, azimuth):
    r"""Angle in degrees between `vector` and the comes-from direction."""
    towards = arrival_direction(zenith, azimuth)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        return np.nan
    return np.degrees(np.arccos(np.clip(vector @ towards / norm, -1.0, 1.0)))


def xmax_above_ground(xmax_pos_shc, zenith, azimuth, ground_altitude):
    r"""Returns Xmax in the shower-core frame, whichever frame the file used.

    Parameters
    ----------
    xmax_pos_shc : array-like, shape (3,)
        As stored, in metres, x North, y West, z Up.
    zenith, azimuth : float
        The shower's stored direction, in degrees, "comes from".
    ground_altitude : float
        The site's ground altitude in metres, ``origin_geoid[2]``.

    Returns
    -------
    numpy.ndarray
        Xmax with ``z`` above the ground, shape (3,).
    str
        ``"ground"`` if the file already stored it so, ``"sea-level"`` if the
        ground altitude was subtracted here, ``"undetermined"`` if neither
        reading follows the stored direction -- then the value is returned as
        stored, with a warning.

    Notes
    -----
    A shower straight down cannot be told apart: both readings point up.
    Then, and whenever both match, the stored value is kept as it is.
    """
    zenith, azimuth = _angles(zenith, azimuth, "xmax_above_ground")
    _validate.as_real(ground_altitude, "ground_altitude", "xmax_above_ground")
    stored = _vector3(xmax_pos_shc, "xmax_pos_shc", "xmax_above_ground")
    shifted = stored - np.array([0.0, 0.0, float(ground_altitude)])
    as_stored = _angle_to_direction(stored, zenith, azimuth)
    as_shifted = _angle_to_direction(shifted, zenith, azimuth)

    if as_stored <= TOLERANCE_DEG:
        return stored, GROUND
    if as_shifted <= TOLERANCE_DEG:
        return shifted, SEA_LEVEL
    # The two angles read like zeniths; they are offsets from the axis (#188)
    logger.warning(
        "xmax_pos_shc %s is not on the shower axis (zenith %.2f, azimuth %.2f deg): it is "
        "%.2f deg off the axis as stored, and %.2f deg off with the ground altitude %.1f m "
        "removed (tolerance %.2f deg). Using it as stored.",
        stored, zenith, azimuth, as_stored, as_shifted, ground_altitude, TOLERANCE_DEG)
    return stored, UNDETERMINED


def xmax_in_site_frame(xmax_pos_shc, zenith, azimuth, ground_altitude, shower_core_pos):
    r"""Returns Xmax in the site frame, the one ``tshower.xmax_pos`` is in.

    The site frame (the "GRAND detector ref" of the data format) is the frame
    of ``du_xyz`` and ``shower_core_pos``: x North, y West, z Up, origin at
    ``origin_geoid``.  Xmax there is its ground-relative shower-core position
    plus the core position -- the point ``get_simu_parameters`` returns as
    ``FIX_xmax_pos``.

    Parameters
    ----------
    xmax_pos_shc : array-like, shape (3,)
        As stored, in metres, in either vertical frame.
    zenith, azimuth : float
        The shower's stored direction, in degrees, "comes from".
    ground_altitude : float
        The ground altitude in metres that a sea-level ``z`` would carry.
    shower_core_pos : array-like, shape (3,)
        The core in the site frame, in metres.

    Returns
    -------
    numpy.ndarray
        Xmax in the site frame, shape (3,).  All NaN if ``xmax_pos_shc`` is
        not finite: a simulation that does not know Xmax says so.
    str
        The frame ``xmax_pos_shc`` was found in, as :func:`xmax_above_ground`
        reports it; ``"undetermined"`` for a non-finite input.
    """
    stored = _vector3(xmax_pos_shc, "xmax_pos_shc", "xmax_in_site_frame")
    core = _vector3(shower_core_pos, "shower_core_pos", "xmax_in_site_frame")
    # Before the unknown-Xmax shortcut, which skipped them (#267)
    _angles(zenith, azimuth, "xmax_in_site_frame")
    _validate.as_real(ground_altitude, "ground_altitude", "xmax_in_site_frame")
    if not np.all(np.isfinite(stored)):
        return np.full(3, np.nan), UNDETERMINED
    above_ground, frame = xmax_above_ground(stored, zenith, azimuth, ground_altitude)
    return above_ground + core, frame
