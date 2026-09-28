# -*- coding: utf-8 -*-
r"""``xmax_above_ground`` tells a file's Xmax frame from its geometry.

The end-to-end cases -- the committed samples and fresh converter output --
are in ``tests/sim2root/test_xmax_frame.py``.  These pin the rule itself.
"""

import numpy as np
import pytest

from grand.dataio.xmax_frame import (GROUND, SEA_LEVEL, UNDETERMINED,
                                     arrival_direction, propagation_direction,
                                     xmax_above_ground, xmax_in_site_frame)

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


CORE = np.array([670.608, -4249.8, 0.0])


@pytest.mark.parametrize('stored_offset', [0.0, GROUND_ALTITUDE])
def test_the_site_frame_adds_the_core_to_the_ground_relative_xmax(stored_offset):
    r"""``tshower.xmax_pos``: same point whichever frame the input was in."""
    truth = _xmax(51.64, 135.44, 7236.0)
    stored = truth + [0.0, 0.0, stored_offset]
    value, frame = xmax_in_site_frame(stored, 51.64, 135.44, GROUND_ALTITUDE, CORE)
    assert frame == (GROUND if stored_offset == 0.0 else SEA_LEVEL)
    assert np.allclose(value, truth + CORE)


def test_the_site_frame_keeps_an_unknown_xmax_unknown():
    r"""NaN in, NaN out, and no zeros that would read as a real position."""
    value, frame = xmax_in_site_frame([np.nan] * 3, 60.0, 30.0, 1142.0, CORE)
    assert frame == UNDETERMINED
    assert np.isnan(value).all()


@pytest.mark.parametrize('zenith, azimuth', [(0.0, 0.0), (51.64, 135.44), (79.43, 310.0)])
def test_propagation_is_opposite_to_arrival(zenith, azimuth):
    r"""The shower travels away from where it comes from: down, for zenith < 90."""
    arrival = arrival_direction(zenith, azimuth)
    travel = propagation_direction(zenith, azimuth)
    assert np.linalg.norm(travel) == pytest.approx(1.0)
    assert np.allclose(travel, -arrival)
    assert travel[2] < 0
    # Xmax lies upstream, on the arrival side.
    assert np.allclose(arrival, _xmax(zenith, azimuth, 1.0))
