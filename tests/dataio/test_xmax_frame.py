# -*- coding: utf-8 -*-
r"""``xmax_above_ground`` tells a file's Xmax frame from its geometry.

The end-to-end cases -- the committed samples and fresh converter output --
are in ``tests/sim2root/test_xmax_frame.py``.  These pin the rule itself.
"""

import numpy as np
import pytest

from grand.dataio.xmax_frame import (GROUND, SEA_LEVEL, UNDETERMINED,
                                     xmax_above_ground)

GROUND_ALTITUDE = 1264.0


def _xmax(zenith, azimuth, distance):
    r"""A ground-relative Xmax on the axis of a shower from (zenith, azimuth)."""
    theta, phi = np.deg2rad(zenith), np.deg2rad(azimuth)
    return distance * np.array([np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi), np.cos(theta)])


@pytest.mark.parametrize('zenith, azimuth, distance',
                         [(51.64, 135.44, 7236.0), (79.43, 310.0, 70000.0),
                          (84.23, 102.47, 148000.0)])
def test_both_frames_come_back_above_the_ground(zenith, azimuth, distance):
    r"""Stored either way, Xmax comes back at the same ground-relative point."""
    truth = _xmax(zenith, azimuth, distance)

    value, frame = xmax_above_ground(truth, zenith, azimuth, GROUND_ALTITUDE)
    assert frame == GROUND
    assert np.allclose(value, truth)

    raised = truth + [0.0, 0.0, GROUND_ALTITUDE]
    value, frame = xmax_above_ground(raised, zenith, azimuth, GROUND_ALTITUDE)
    assert frame == SEA_LEVEL
    assert np.allclose(value, truth)


def test_a_position_off_the_axis_is_left_as_stored():
    r"""Neither reading on the axis: kept as stored, and said so.

    The synthetic showers in the pipeline tests store Xmax straight up at
    zenith 85; they must keep reading exactly as they did.
    """
    stored = [0.0, 0.0, 10000.0]
    value, frame = xmax_above_ground(stored, 85.0, 0.0, 1200.0)
    assert frame == UNDETERMINED
    assert np.array_equal(value, stored)


def test_nan_is_left_as_stored():
    r"""The CoREAS converter writes NaN when it has no Xmax position."""
    value, frame = xmax_above_ground([np.nan] * 3, 60.0, 30.0, 1142.0)
    assert frame == UNDETERMINED
    assert np.isnan(value).all()


def test_a_vertical_shower_is_left_as_stored():
    r"""Straight down, both readings point up; nothing to decide from."""
    stored = [0.0, 0.0, 5000.0]
    value, frame = xmax_above_ground(stored, 0.0, 0.0, GROUND_ALTITUDE)
    assert frame == GROUND
    assert np.array_equal(value, stored)
