# -*- coding: utf-8 -*-
# Copyright (C) 2019 The GRAND collaboration
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

"""Topography wrapper for GRAND packages.
"""

from typing import Optional, Union
from typing_extensions import Final

from . import DATADIR
from .coordinates import ECEF, GeodeticRepresentation, LTP
from ..libs.turtle import Stack as _Stack, Map as _Map
from .. import store

import os
from pathlib import Path

import astropy.units as u
import numpy

__all__ = ["elevation", "geoid_undulation", "update_data", "cachedir",
           "model", "Topography"]


_CACHEDIR: Final = Path(__file__).parent / "data" / "topography"
"""Location of cached topography data"""


_DEFAULT_MODEL: Final = "SRTMGL1"
"""The default topographic model"""


_default_topography: Optional["Topography"] = None
"""Stack for the topographic data"""


_geoid: Optional[_Map] = None
"""Map with geoid undulations"""


def elevation(coordinates: Union[ECEF, LTP]) -> u.Quantity:
    """Get the topography elevation, w.r.t. sea level.
    """
    global _default_topography

    if _default_topography is None:
        _CACHEDIR.mkdir(exist_ok=True)
        _default_topography = Topography(_CACHEDIR)
    return _default_topography.elevation(coordinates)


def geoid_undulation(coordinates: Union[ECEF, LTP]) -> u.Quantity:
    """Get the geoid undulation.
    """

    global _geoid

    if _geoid is None:
        path = os.path.join(DATADIR, "egm96.png")
        _geoid = _Map(path)

    # Compute the geodetic coordinates
    cart = coordinates.transform_to(ECEF).cartesian
    geodetic = cart.represent_as(GeodeticRepresentation)

    z = _geoid.elevation(geodetic.longitude / u.deg,
                         geodetic.latitude / u.deg)
    return z * u.m


def update_data(coordinates: Union[ECEF, LTP]=None, clear: bool=False,
                radius: u.Quantity=None):
    """Update the cache of topography data.
    """
    if clear:
        for p in _CACHEDIR.glob("**/*.*"):
            p.unlink()

    if coordinates is not None:
        _CACHEDIR.mkdir(exist_ok=True)

        # Compute the bounding box
        coordinates = coordinates.transform_to(ECEF)
        coordinates = coordinates.represent_as(GeodeticRepresentation) # type:ignore
        latitude = coordinates.latitude / u.deg # type: ignore
        try:
            latitude = [min(latitude), max(latitude)]
        except TypeError:
            latitude = [latitude, latitude]
        longitude = coordinates.longitude / u.deg # type: ignore
        try:
            longitude = [min(longitude), max(longitude)]
        except TypeError:
            longitude = [longitude, longitude]

        # Extend by the radius, if any
        if radius is not None:
            for i in range (2):
                delta = -radius if not i else radius
                c = LTP(x=delta, y=delta, z=0 * u.m,
                        location=ECEF(GeodeticRepresentation(
                                          latitude[i] * u.deg,
                                          longitude[i] * u.deg)))
                c = c.transform_to(ECEF).represent_as(GeodeticRepresentation)
                latitude[i] = c.latitude / u.deg
                longitude[i] = c.longitude / u.deg


        # Get the corresponding tiles
        longitude = [int(numpy.floor(lon)) for lon in longitude]
        latitude = [int(numpy.floor(lat)) for lat in latitude]

        for lat in range(latitude[0], latitude[1] + 1):
            for lon in range(longitude[0], longitude[1] + 1):
                if lat < 0:
                    ns = "S"
                    lat = -lat
                else:
                    ns = "N"

                if lon < 0:
                    ew = "W"
                    lon = -lon
                else:
                    ew = "E"

                basename = f"{ns}{lat:02.0f}{ew}{lon:03.0f}.SRTMGL1.hgt"
                path = _CACHEDIR / basename
                if not path.exists():
                    with path.open("wb") as f:
                        f.write(store.get(basename))


def cachedir() -> Path:
    """Get the location of the topography data cache.
    """
    return _CACHEDIR


def model() -> str:
    """Get the default model for topographic data.
    """
    return _DEFAULT_MODEL


class Topography:
    """Proxy to topography data.
    """

    def __init__(self, path: Union[Path, str]) -> None:
        self._stack = _Stack(str(path))


    def elevation(self, coordinates: Union[ECEF, LTP]) -> u.Quantity:
        """Get the topography elevation, w.r.t. sea level.
        """
        # Compute the geodetic coordinates
        cart = coordinates.transform_to(ECEF).cartesian
        geodetic = cart.represent_as(GeodeticRepresentation)

        # Return the topography elevation
        z = self._stack.elevation(geodetic.latitude / u.deg,
                                  geodetic.longitude / u.deg)
        return z * u.m
