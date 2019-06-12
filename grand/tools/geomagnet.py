# -*- coding: utf-8 -*-
# Copyright (C) 2019 The GRAND collaboration
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

"""Geomagnetic field wrapper for GRAND packages
"""

from typing import Union

from .coordinates import ECEF, GeodeticRepresentation, LTP
from ..libs.gull import Snapshot as _Snapshot

import astropy.units as u
import numpy

from astropy.coordinates import EarthLocation


_DEFAULT_MODEL = "IGRF12" # type: str
"""The default geo-magnetic model"""


_default_magnet = None # type: Geomagnet
"""An instance of Geomagnet with the default geo-magnetic model"""


def field(coordinates: Union[ECEF, LTP]) -> Union[ECEF, LTP]:
    """Get the default geo-magnetic field at the given *coordinates*.
    """
    global _default_magnet

    if _default_magnet is None:
        _default_magnet = Geomagnet()
    return _default_magnet.field(coordinates)


def model() -> str:
    """Get the default model for the geo-magnetic field, I.e.
    `IGRF12 <https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html>`_.
    """
    return _DEFAULT_MODEL


class Geomagnet:
    """Proxy to a geomagnetic model
    """

    def __init__(self, model: str=None):
        if model is None:
            model = _DEFAULT_MODEL
        self._model = model
        self._snapshot = None
        self._date = None


    def field(self, coordinates: Union[ECEF, LTP]) -> Union[ECEF, LTP]:
        """Get the geo-magnetic field components
        """

        # Update the snapshot, if needed
        obstime = coordinates.obstime
        if obstime is None:
            raise ValueError(
                "No observation time was specified for the coordinates")
        date = obstime.datetime.date()
        if date != self._date:
            self._snapshot = _Snapshot(self._model, date)
            self._date = date

        # Compute the geodetic coordinates
        cart = coordinates.transform_to(ECEF).cartesian
        geodetic = cart.represent_as(GeodeticRepresentation)

        # Fetch the magnetic field components in local LTP
        field = self._snapshot(geodetic.latitude / u.deg,
                               geodetic.longitude / u.deg,
                               geodetic.height / u.m)

        # Encapsulate the result
        n = geodetic.latitude.size
        if n == 1:
            location = EarthLocation(lat=geodetic.latitude,
                                     lon=geodetic.longitude,
                                     height=geodetic.height)
            return LTP(x=field[0] * u.T, y=field[1] * u.T, z=field[2] * u.T,
                       location=location, obstime=obstime)
        else:
            ecef = numpy.zeros((n, 3))
            for i, value in enumerate(field):
                location = EarthLocation(lat=geodetic.latitude[i],
                                         lon=geodetic.longitude[i],
                                         height=geodetic.height[i])
                ltp = LTP(x=value[0], y=value[1], z=value[2], location=location)
                c = ltp.transform_to(ECEF).cartesian
                ecef[i,0] = c.x
                ecef[i,1] = c.y
                ecef[i,2] = c.z
            return ECEF(x=ecef[:,0] * u.T, y=ecef[:,1] * u.T, z=ecef[:,2] * u.T,
                        obstime=obstime)
