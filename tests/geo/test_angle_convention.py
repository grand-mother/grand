# -*- coding: utf-8 -*-
r"""GRAND's angles name the direction a shower comes *from*.

Decided 2026-09-24 (resources/dev/dev-next/DECISIONS.md, section 3), after
``snonis_sim2root_test_merge`` proposed flipping the core transforms in
``grand/geo/coordinates.py`` to the direction the shower travels
(zenith -> 180 - zenith, azimuth -> azimuth + 180).  The flip was not taken,
because everything GRANDlib reads already uses "comes from": the simulation
files, the converters and the paper's Appendix A.

This pins the convention against an oracle outside GRANDlib -- the ZHAireS
summary files committed with the example events.  Each ``.sry`` states the
primary's zenith and azimuth, and separately the Cartesian position of the
shower maximum relative to the core.  Xmax lies upstream on the shower axis,
so converting its position to angles must give back the stated direction.
A flipped convention returns the opposite direction and fails here.
"""

import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRY = sorted((ROOT / 'sim2root' / 'ZHAireSRawRoot').glob('*/*.sry'))

needs_sry = pytest.mark.skipif(not SRY, reason='no ZHAireS .sry fixtures')


def _read(path):
    r"""Returns the stated direction and the ground-relative Xmax of a summary.

    Parameters
    ----------
    path : pathlib.Path
        A ZHAireS ``.sry`` file.

    Returns
    -------
    tuple
        ``(zenith_deg, azimuth_deg, (x, y, z))``, the position in metres from
        the core, with ``z`` measured from the ground.
    """
    text = path.read_text(encoding='utf-8', errors='replace')

    def number(pattern):
        found = re.search(pattern, text)
        assert found, '%r not found in %s' % (pattern, path.name)
        return float(found.group(1))

    zenith = number(r'Primary zenith angle:\s+([-0-9.]+)\s+deg')
    azimuth = number(r'Primary azimuth angle:\s+([-0-9.]+)\s+deg')
    ground = number(r'Ground altitude:\s+([-0-9.eE+]+)\s+km') * 1000.0
    pos = re.search(r'Pos\. Max\.:' + r'\s+(-?[0-9.eE+-]+)' * 5, text)
    assert pos, 'no "Pos. Max." line in %s' % path.name
    x, y, z = (float(pos.group(i)) * 1000.0 for i in (3, 4, 5))
    assert 'Local magnetic north' in text, (
        '%s does not measure azimuth from magnetic north; the comparison '
        'below assumes it does' % path.name)
    return zenith, azimuth, (x, y, z - ground)


@needs_sry
@pytest.mark.parametrize('sry', SRY, ids=[p.parent.name[-5:] for p in SRY])
def test_xmax_position_gives_back_the_stated_arrival_direction(sry):
    r"""The core spherical transform maps Xmax to the file's own angles."""
    from grand.geo.coordinates import _cartesian_to_spherical

    zenith, azimuth, (x, y, z) = _read(sry)
    theta, phi, _ = _cartesian_to_spherical(x, y, z)
    theta, phi = float(np.ravel(theta)[0]), float(np.ravel(phi)[0]) % 360.0

    # The .sry prints angles to two decimals and positions to ten metres.
    assert theta == pytest.approx(zenith, abs=0.02), (
        'zenith %.2f from Xmax, %.2f stated: if this is 180 minus the stated '
        'value, the convention has been flipped to "travels towards"'
        % (theta, zenith))
    assert phi == pytest.approx(azimuth % 360.0, abs=0.02), (
        'azimuth %.2f from Xmax, %.2f stated: 180 degrees off means the '
        'convention has been flipped' % (phi, azimuth))


def test_the_transform_accepts_arrays():
    r"""The flipped version raised on arrays (``if phi==360``); this must not."""
    from grand.geo.coordinates import _cartesian_to_spherical

    theta, phi, r = _cartesian_to_spherical(np.array([1.0, 0.0]),
                                            np.array([0.0, 1.0]),
                                            np.array([1.0, 1.0]))
    assert np.allclose(theta, 45.0) and np.allclose(phi, [0.0, 90.0])
