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
    assert "self.antennas_origin = origin_geoid" in source
    assert Event()._gps_origin() == GPS_ANTENNA_ORIGIN       # the default


GP80 = ROOT / "examples" / "analysis" / "reconstructed_events_AOI"


@pytest.mark.skipif(not GP80.is_dir(), reason="the GP80 example is not present")
def test_the_gps_origin_can_be_chosen():
    r"""#215: GPS positions against the fixed point (default), the run's origin, or a given one."""
    from grand.aoi.event import GPS_ANTENNA_ORIGIN
    from grand.aoi.event_list import EventList

    def first(**options):
        event = next(iter(EventList(str(GP80), **options)))
        positions = {a.id: np.ravel(np.asarray(a.position)) for a in event.antennas}
        return event, positions

    fixed, at_fixed = first()
    assert fixed.antennas_origin == GPS_ANTENNA_ORIGIN
    run, at_run = first(gps_origin="run")
    origin = tuple(np.ravel(np.asarray(run.trun.origin_geoid)))
    assert run.antennas_origin == pytest.approx(origin)
    # The two origins lie ~3.8 km apart, so every position moves by about that
    shift = [np.hypot(*(at_fixed[k] - at_run[k])[:2]) for k in at_fixed]
    assert 3000 < min(shift) and max(shift) < 4500
    given, _ = first(gps_origin=origin)
    assert given.antennas_origin == pytest.approx(origin)
    with pytest.raises(ValueError, match="Event.gps_origin: must be None"):
        first(gps_origin="centre")
