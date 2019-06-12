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

"""Extra frames for astropy.coordinates.
"""

from typing import Sequence, Union
from datetime import datetime

from .representation import GeodeticRepresentation, HorizontalRepresentation

from ...libs import turtle

import numpy
import astropy.units as u
from astropy.coordinates import Attribute, BaseCoordinateFrame,                \
                                CartesianRepresentation,                       \
                                CylindricalRepresentation, EarthLocation,      \
                                FunctionTransform, ITRS,                       \
                                PhysicsSphericalRepresentation,                \
                                RepresentationMapping, TimeAttribute,          \
                                frame_transform_graph
from astropy.time import Time


__all__ = ["ECEF", "LTP"]


_HAS_GEOMAGNET = False
"""The geomagnet module needs a deferred import due to circular references."""


class ECEF(BaseCoordinateFrame):
    """Earth-Centered Earth-Fixed frame, co-moving with the Earth.
    """

    default_representation = CartesianRepresentation
    """Default representation of local frames."""

    obstime = TimeAttribute(default=None)
    """The observation time. Defaults to `None`."""


    def __init__(self, *args, obstime: Union[datetime, Time, str]=None,
                 **kwargs):
        super().__init__(*args, obstime=obstime, **kwargs)


    @property
    def earth_location(self) -> EarthLocation:
        """The data in this frame as an
        :py:class:`~astropy.coordinates.EarthLocation`.
        """
        geo = self.represent_as(GeodeticRepresentation)
        return EarthLocation(lon=geo.longitude, lat=geo.latitude,
                             height=geo.height)


@frame_transform_graph.transform(FunctionTransform, ITRS, ECEF)
def itrs_to_ecef(itrs: ITRS, ecef: ECEF) -> ECEF:
    """Compute the transformation from ITRS to ECEF coordinates.
    """
    if ecef._obstime == None:
        # The conversion goes to a generic frame
        ecef._obstime = itrs._obstime
    elif ecef._obstime != itrs._obstime:
        itrs = itrs.transform_to(ITRS(obstime=ecef._obstime))

    return ecef.realize_frame(itrs.cartesian)


@frame_transform_graph.transform(FunctionTransform, ECEF, ITRS)
def ecef_to_itrs(ecef: ECEF, itrs: ITRS) -> ITRS:
    """Compute the transformation from ECEF to ITRS coordinates.
    """
    if itrs._obstime == ecef._obstime:
        c = ecef.cartesian
    else:
        itrs0 = ITRS(ecef.cartesian, obstime=ecef._obstime)
        c = itrs0.transform_to(ITRS(obstime=itrs._obstime)).cartesian

    return itrs.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, ECEF, ECEF)
def ecef_to_ecef(ecef0: ECEF, ecef1: ECEF) -> ECEF:
    """Compute the transformation from ECEF to ECEF coordinate.
    """
    if ecef1._obstime is None:
        ecef1._obstime = ecef0._obstime

    return ecef1.realize_frame(ecef0.cartesian)


class LTP(BaseCoordinateFrame): # Forward declaration
    pass


class LTP(BaseCoordinateFrame):
    """Local frame tangent to the WGS84 ellipsoid and oriented along cardinal
       directions.
    """

    default_representation = CartesianRepresentation
    """Default representation of local frames."""

    location = Attribute(default=None)
    """The origin on Earth of the local frame."""

    orientation = Attribute(default=("E", "N", "U"))
    """The cardinal directions of the x, y, and z axis (default: E, N, U)."""

    magnetic = Attribute(default=False)
    """Use the magnetic north instead of the geographic one (default: false)."""

    obstime = TimeAttribute(default=None)
    """The observation time."""


    def __init__(self, *args, location: Union[EarthLocation, ECEF, LTP]=None,
                 orientation: Sequence[str]=None, magnetic: bool=False,
                 obstime: Union[datetime, Time, str]=None, **kwargs):

        # Do the base initialisation
        location = self.location if location is None else location
        if hasattr(location, "earth_location"):
            location = location.earth_location
        orientation = self.orientation if orientation is None else orientation

        super().__init__(*args, location=location, orientation=orientation,
                         magnetic=magnetic, obstime=obstime, **kwargs)

        # Set the transform parameters
        itrs = self._location.itrs
        geo = itrs.represent_as(GeodeticRepresentation)
        latitude, longitude = geo.latitude / u.deg, geo.longitude / u.deg

        if magnetic:
            # Compute the magnetic declination
            if self._obstime is None:
                raise ValueError("Magnetic coordinates require specifying "
                                 "an observation time")
            ecef = ECEF(itrs.x, itrs.y, itrs.z, obstime=self._obstime)

            if not _HAS_GEOMAGNET:
                from ..geomagnet import field as _geomagnetic_field
            field = _geomagnetic_field(ecef)

            c = field.cartesian
            c /= c.norm()
            h = c.represent_as(HorizontalRepresentation)
            azimuth0 = h.azimuth / u.deg
        else:
            azimuth0 = 0.

        def vector(name):
            tag = name[0].upper()
            if tag == "E":
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   90 + azimuth0, 0)
            elif tag == "W":
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   270 + azimuth0, 0)
            elif tag == "N":
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   azimuth0,  0)
            elif tag == "S":
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   180 + azimuth0,  0)
            elif tag == "U":
                return turtle.ecef_from_horizontal(latitude, longitude, 0, 90)
            elif tag == "D":
                return turtle.ecef_from_horizontal(latitude, longitude, 0, -90)
            else:
                raise ValueError(f"Invalid frame orientation `{name}`")

        ux = vector(self._orientation[0])
        uy = vector(self._orientation[1])
        uz = vector(self._orientation[2])

        self._basis = numpy.column_stack((ux, uy, uz))
        self._origin = itrs.cartesian


    @property
    def earth_location(self) -> EarthLocation:
        """The data in this frame as an
        :class:`~astropy.coordinates.EarthLocation`.
        """
        geo = self.transform_to(ECEF).represent_as(GeodeticRepresentation)
        return EarthLocation(lon=geo.longitude, lat=geo.latitude,
                             height=geo.height)


@frame_transform_graph.transform(FunctionTransform, ECEF, LTP)
def ltp_to_ltp(ecef: ECEF, ltp: LTP) -> LTP:
    """Compute the transformation from ECEF to LTP coordinates.
    """
    c = ecef.cartesian
    if c.x.unit.is_equivalent("m"):
        c = c.copy()
        c -= ltp._origin
    c = c.transform(ltp._basis.T)

    if ltp._obstime is None:
        ltp._obstime = ecef._obstime

    return ltp.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, LTP, ECEF)
def ltp_to_ecef(ltp: LTP, ecef: ECEF) -> ECEF:
    """Compute the transformation from LTP to ECEF coordinates.
    """
    c = ltp.cartesian.transform(ltp._basis)
    if c.x.unit.is_equivalent("m"):
        c += ltp._origin

    if ecef._obstime is None:
        ecef._obstime = ltp._obstime

    return ecef.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, LTP, LTP)
def ltp_to_ltp(ltp0: LTP, ltp1: LTP) -> LTP:
    """Compute the transformation from LTP to LTP coordinates.
    """
    c = ltp0.cartesian
    translate = c.x.unit.is_equivalent("m")

    # Forward the observation time
    if ltp1._obstime is None:
        ltp1._obstime = ltp0._obstime

    # Check if the two frames are identicals
    if numpy.array_equal(ltp0._basis, ltp1._basis):
        if not translate or ((ltp0._origin.x == ltp1._origin.x) and
                             (ltp0._origin.y == ltp1._origin.y) and
                             (ltp0._origin.z == ltp1._origin.z)):
            # CartesianRepresentations might not evaluate to equal though the
            # coordinates are equal
            return ltp1.realize_frame(c)

    # Transform to ECEF
    if translate:
        c = c.copy()
        c -= ltp0._origin
    c = c.transform(ltp0._basis.T)

    # Transform back from ECEF
    c = c.transform(ltp1._basis)
    if translate:
        c += ltp1._origin

    return ltp1.realize_frame(c)
