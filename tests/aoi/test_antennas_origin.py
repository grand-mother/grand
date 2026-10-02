# -*- coding: utf-8 -*-
r"""#215: an event says which origin its antenna positions are relative to.

GPS-derived positions (GP300, GP80, GP13) used an unnamed hard-coded origin,
3.8 km from the run's ``origin_geoid``, and nothing on the event said so.
"""

import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="the committed sample is not present")


def _event():
    from grand.aoi.event_list import EventList
    return next(iter(EventList(str(SAMPLE))))


def test_positions_from_the_run_tree_use_its_origin():
    event = _event()
    assert event.antennas
    assert event.antennas_origin == pytest.approx(tuple(np.ravel(np.asarray(event.trun.origin_geoid))))


def test_positions_from_gps_use_the_named_origin():
    # Simulated files carry no GPS branches, so the GPS path cannot run here:
    # this checks that it reads the named constant and records it.
    import inspect

    from grand.aoi.event import GPS_ANTENNA_ORIGIN, Event

    assert GPS_ANTENNA_ORIGIN == (40.95068711, 93.96977396, 1200.0)
    source = inspect.getsource(Event.fill_antennas)
    assert "40.95068711" not in source
    assert "= GPS_ANTENNA_ORIGIN" in source and "self.antennas_origin = GPS_ANTENNA_ORIGIN" in source
