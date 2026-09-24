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

logger = logging.getLogger(__name__)

#: How closely the core-to-Xmax vector must follow the stored direction, in
#: degrees.  Measured agreement is 0.001; the smallest miss is 0.49.
TOLERANCE_DEG = 0.05

GROUND = "ground"
SEA_LEVEL = "sea-level"
UNDETERMINED = "undetermined"


def _angle_to_direction(vector, zenith, azimuth):
    r"""Angle in degrees between `vector` and the comes-from direction."""
    theta, phi = np.deg2rad(zenith), np.deg2rad(azimuth)
    towards = np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)])
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
    stored = np.asarray(xmax_pos_shc, dtype=float).reshape(3)
    shifted = stored - np.array([0.0, 0.0, float(ground_altitude)])
    as_stored = _angle_to_direction(stored, zenith, azimuth)
    as_shifted = _angle_to_direction(shifted, zenith, azimuth)

    if as_stored <= TOLERANCE_DEG:
        return stored, GROUND
    if as_shifted <= TOLERANCE_DEG:
        return shifted, SEA_LEVEL
    logger.warning(
        "xmax_pos_shc %s follows neither reading of the shower direction "
        "(zenith %.2f, azimuth %.2f): %.2f deg as stored, %.2f deg with the "
        "ground altitude %.1f m removed. Using it as stored.",
        stored, zenith, azimuth, as_stored, as_shifted, ground_altitude)
    return stored, UNDETERMINED
