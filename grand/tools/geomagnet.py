"""Geomagnetic field wrapper for GRAND packages
"""
from __future__ import annotations

from typing import Optional, Union
from typing_extensions import Final

from .coordinates import CartesianRepresentation, GeodeticRepresentation
from .coordinates import ECEF, Geodetic, GRANDCS, LTP, _cartesian_to_horizontal
from ..libs.gull import Snapshot as _Snapshot

import numpy
import datetime
from numbers import Number


_default_model: Final = "IGRF13"
"""The default geo-magnetic model, i.e. IGRF13.
   Reference: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
"""

_default_magnet: Optional[Geomagnet] = None
"""An instance of Geomagnet with the default geo-magnetic model"""

# _default_obstime: Final[Time] = Time('2020-01-01')
_default_obstime: Final[datetime.date] = datetime.date(2020, 1, 1)
"""The default observation time if none is specified"""


def __getattr__(name):
    if name == "model":
        return _default_model
    elif name == "obstime":
        # return _default_obstime.datetime.date
        return _default_obstime
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# This function is no longer necessary. Might show up in other part of the grandlib.
def field(coordinates: Union[ECEF, Geodetic, GRANDCS, LTP]) -> CartesianRepresentation:
    """Get the default geo-magnetic field at the given *coordinates*."""
    # global _default_magnet
    # if _default_magnet is None:
    #    #_default_magnet = Geomagnet()
    # return _default_magnet.field(coordinates)
    geomagnet = Geomagnet(location=coordinates)
    return geomagnet.field


class Geomagnet:
    """Proxy to a geomagnetic model. 'IGRF13' is used as a default model.
    Get the geo-magnetic field components [Bx, By, Bz] on any geodetic location of
    Earth at any given time. For location, either provide latitude (deg), longitude (deg), and
    height (m) or provide location in ECEF, Geodetic, or GRAND coordinate system. TypeError
    will occur if location is not provided. For obstime, provide time in isoformat
    ('2020-01-19') or in datetime.date(2020, 1, 19). If obstime is not provided, a
    default value ('2020-01-01') is used.
    """

    def __init__(
        self,
        model: str = None,
        latitude: Union[Number, numpy.ndarray] = None,
        longitude: Union[Number, numpy.ndarray] = None,
        height: Union[Number, numpy.ndarray] = None,
        location: Union[ECEF, Geodetic, LTP, GRANDCS] = None,
        obstime: Union[str, datetime.date] = None,
    ) -> CartesianRepresentation:

        # print('location:', location, type(location))
        if model is None:
            model = _default_model

        # Make sure time is in isoformat of datetime.date() format.
        if obstime is None:
            obstime = _default_obstime
        elif isinstance(obstime, (str, datetime.date)):
            pass
        else:
            raise TypeError(
                "obstime given is of type %s. Provide obstime in string or datetime.date type.\
                Example: '2020-01-19' or datetime.date(2020, 1, 19)."
                % type(obstime)
            )

        # Make sure the location is in the correct format. i.e ECEF, Geodetic, GeodeticRepresentation,
        # or GRAND cs. OR latitude=deg, longitude=deg, height=meter.
        if latitude != None and longitude != None and height != None:
            geodetic_loc = Geodetic(
                latitude=latitude, longitude=longitude, height=height
            )
        elif isinstance(
            location, (ECEF, Geodetic, GeodeticRepresentation, LTP, GRANDCS)
        ):
            geodetic_loc = Geodetic(location)
        else:
            raise TypeError(
                "Provide location in ECEF, Geodetic, or GRAND coordinate system instead of type %s.\n \
                            Location can also be given as latitude=deg, longitude=deg, height=meter."
                % type(location)
            )

        self.model = model  # type: str
        self.obstime = obstime
        self.location = geodetic_loc

        # Calculate magnetic field
        self.snapshot = _Snapshot(self.model, self.obstime)
        Bfield = self.snapshot(
            geodetic_loc.latitude, geodetic_loc.longitude, geodetic_loc.height
        )

        # Output magnetic field is either in [Bx, By, Bz] or [[Bx1, By1, Bz1], [Bx2, By2, Bz2], ....]
        if Bfield.size == 3:
            Bx, By, Bz = Bfield[0], Bfield[1], Bfield[2]
        elif Bfield.size > 3:
            Bx, By, Bz = Bfield[:, 0], Bfield[:, 1], Bfield[:, 2]

        self.field = CartesianRepresentation(x=Bx, y=By, z=Bz)

        # calculate magnetic declination and inclination
        azimuth, elevation, norm = _cartesian_to_horizontal(Bx, By, Bz)
        self.declination = azimuth
        self.inclination = -elevation
