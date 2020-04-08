'''Geomagnetic field wrapper for GRAND packages
'''
from __future__ import annotations

from typing import Optional, Union
from typing_extensions import Final

from .coordinates import ECEF, GeodeticRepresentation, LTP
from ..libs.gull import Snapshot as _Snapshot

import astropy.units as u
from astropy.time import Time
import numpy

from astropy.coordinates import EarthLocation


_default_model: Final = 'IGRF13'
'''The default geo-magnetic model, i.e. IGRF13.
   Reference: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
'''

_default_magnet: Optional[Geomagnet] = None
'''An instance of Geomagnet with the default geo-magnetic model'''

_default_obstime: Final[Time] = Time('2020-01-01')
'''The default observation time if none is specified'''


def __getattr__(name):
    if name == 'model':
        return _default_model
    elif name == 'obstime':
        return _default_obstime.datetime.date
    else:
        raise AttributeError(f'module {__name__} has no attribute {name}')


def field(coordinates: Union[ECEF, LTP]) -> Union[ECEF, LTP]:
    '''Get the default geo-magnetic field at the given *coordinates*.
    '''
    global _default_magnet

    if _default_magnet is None:
        _default_magnet = Geomagnet()
    return _default_magnet.field(coordinates)


class Geomagnet:
    '''Proxy to a geomagnetic model
    '''

    def __init__(self, model: str=None) -> None:
        if model is None:
            model = _default_model
        self._model = model   # type: str
        self._snapshot = None # type: Optional[_Snapshot]
        self._date = None     # type: Optional[str]


    def field(self, coordinates: Union[ECEF, LTP]) -> Union[ECEF, LTP]:
        '''Get the geo-magnetic field components
        '''

        # Update the snapshot, if needed
        obstime = coordinates.obstime
        if obstime is None:
            obstime = _default_obstime
        date = obstime.datetime.date()
        if date != self._date:
            self._snapshot = _Snapshot(self._model, date)
            self._date = date

        # Compute the geodetic coordinates
        cart = coordinates.transform_to(ECEF).cartesian
        geodetic = cart.represent_as(GeodeticRepresentation)

        # Fetch the magnetic field components in local LTP
        field = self._snapshot(geodetic.latitude / u.deg,        # type: ignore
                               geodetic.longitude / u.deg,
                               geodetic.height / u.m)

        # Encapsulate the result
        n = geodetic.latitude.size
        if n == 1:
            location = EarthLocation(lat=geodetic.latitude,
                                     lon=geodetic.longitude,
                                     height=geodetic.height)
            return LTP(x=field[0] * u.T, y=field[1] * u.T, z=field[2] * u.T,
                       location=location, obstime=obstime, orientation='ENU',
                       magnetic=False)
        else:
            ecef = numpy.zeros((n, 3))
            for i, value in enumerate(field):
                location = EarthLocation(lat=geodetic.latitude[i],
                                         lon=geodetic.longitude[i],
                                         height=geodetic.height[i])
                ltp = LTP(x=value[0], y=value[1], z=value[2], location=location,
                          orientation='ENU', magnetic=False)
                c = ltp.transform_to(ECEF).cartesian
                ecef[i,0] = c.x
                ecef[i,1] = c.y
                ecef[i,2] = c.z
            return ECEF(x=ecef[:,0] * u.T, y=ecef[:,1] * u.T, z=ecef[:,2] * u.T,
                        obstime=obstime)
