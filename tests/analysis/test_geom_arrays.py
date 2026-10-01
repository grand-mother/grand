# -*- coding: utf-8 -*-
r"""Geometry helpers on arrays of directions."""

import numpy as np
import pytest


def test_sin_geomag_angle_is_per_direction():
    r"""#214: an array of angles gave one number, above 1."""
    from grand.analysis.geom.angles import sin_geomag_angle

    theta, phi = np.array([0.5, 1.0, 1.4]), np.array([1.0, 2.0, -0.3])
    together = sin_geomag_angle(theta, phi)
    assert together.shape == (3,)
    assert together == pytest.approx([sin_geomag_angle(t, p) for t, p in zip(theta, phi)])
    assert isinstance(sin_geomag_angle(0.5, 1.0), float)
