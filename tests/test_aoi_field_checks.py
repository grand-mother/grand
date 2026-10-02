"""The grand.aoi classes and ShowerEvent check their fields as the trees do (#267)."""
import warnings

import numpy as np
import pytest

from grand.basis.validate import GRANDlibWarning


def _quiet(make):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return make()


def test_shower_fields():
    from grand.aoi.shower import Shower

    shower = _quiet(Shower)
    _quiet(lambda: setattr(shower, "zenith", np.float32(45)))      # tree values pass
    assert isinstance(shower.zenith, np.float32)                     # and keep their type
    _quiet(lambda: setattr(shower, "Xmax", np.nan))                  # unknown
    with pytest.raises(TypeError, match="Shower: 'zenith' must be a real number"):
        shower.zenith = "45"
    with pytest.raises(TypeError, match="boolean|real number"):
        shower.azimuth = True
    with pytest.warns(GRANDlibWarning, match="Shower.azimuth: should be between 0 and 360 degrees, got 400"):
        shower.azimuth = 400
    assert shower.azimuth == 400                                     # stored as given, as the trees do
    with pytest.warns(GRANDlibWarning, match="energy_primary: should be >= 0 GeV"):
        Shower(energy_primary=-1)
    with pytest.raises(TypeError, match="Shower.primary_type: must be a string"):
        shower.primary_type = 2212
    with pytest.raises(ValueError, match="magnetic_field: must hold 3 values"):
        shower.magnetic_field = [1, 2]


def test_ids_and_numbers():
    from grand.aoi.antenna import Antenna
    from grand.aoi.event import Event
    from grand.aoi.timetrace import Efield

    assert _quiet(Antenna).id == -1                                  # default "not set"
    with pytest.raises(TypeError, match="'id' must be an integer, got 1.5"):
        Antenna(id=1.5)
    with pytest.raises(ValueError, match="Antenna.id: must be >= -1, got -5"):
        Antenna(id=-5)
    trace = _quiet(Efield)
    with pytest.raises(ValueError, match="n_points: must be >= 0, got -1"):
        trace.n_points = -1
    with pytest.warns(GRANDlibWarning, match="t_bin_size: should be >= 0 ns"):
        trace.t_bin_size = -2
    event = _quiet(Event)
    event.event_number = np.uint32(1618)
    assert event.event_number == 1618
    with pytest.raises(ValueError, match="Event.run_number: must be >= 0, got -1"):
        event.run_number = -1
    with pytest.raises(TypeError, match="'tefield_level' must be an integer"):
        event.tefield_level = "1"


def test_shower_event_fields():
    from grand import ShowerEvent
    from grand.sim.shower.pdg import ParticleCode

    shower = _quiet(ShowerEvent)
    _quiet(lambda: setattr(shower, "primary", "proton"))
    _quiet(lambda: setattr(shower, "primary", ParticleCode.PROTON))
    with pytest.raises(TypeError, match="ShowerEvent.primary: must be a particle name or a ParticleCode"):
        shower.primary = 2212.5
    with pytest.warns(GRANDlibWarning, match="ShowerEvent.zenith: should be between 0 and 180"):
        shower.zenith = 200
    with pytest.warns(GRANDlibWarning, match=r"origin_geoid\[0\] \(latitude\): should be between -90 and 90"):
        shower.origin_geoid = [95, 93.95, 1200]
    with pytest.raises(ValueError, match="origin_geoid: must hold 3 values"):
        shower.origin_geoid = [40.98, 93.95]
