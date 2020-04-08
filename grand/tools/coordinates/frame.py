'''Extra frames for astropy.coordinates.
'''
from __future__ import annotations

from collections import defaultdict
import copy as _copy
from datetime import datetime
from typing import Any, Optional, Sequence, Union
from typing_extensions import Final

import numpy
import astropy.units as u
from astropy.coordinates import Attribute, BaseCoordinateFrame,                \
                                BaseRepresentation, CartesianRepresentation,   \
                                CylindricalRepresentation, EarthLocation,      \
                                FunctionTransform, ITRS,                       \
                                PhysicsSphericalRepresentation,                \
                                RepresentationMapping, TimeAttribute,          \
                                frame_transform_graph
from astropy.time import Time
from astropy.utils.decorators import lazyproperty

from ...libs import turtle
from .representation import GeodeticRepresentation, HorizontalRepresentation
from .transform import Rotation


__all__ = ['ECEF', 'LTP', 'ExtendedCoordinateFrame']


_HAS_GEOMAGNET: Final = False
'''The geomagnet module needs a deferred import due to circular references.'''


class ExtendedCoordinateFrame(BaseCoordinateFrame):
    '''A coordinates frame with immutable data supporting extra arithmetic
       operators.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._data is not None:
            self._protect(self._data)


    def __add__(self, other: Union[BaseRepresentation, BaseCoordinateFrame])   \
        -> ExtendedCoordinateFrame:
        '''Left hand side coordinates translation.
        '''
        if isinstance(other, BaseCoordinateFrame):
            other = other.transform_to(self)._data
        return self.realize_frame(self._data + other)


    def __radd__(self, other: Union[BaseRepresentation, BaseCoordinateFrame])  \
        -> ExtendedCoordinateFrame:
        '''Right hand side coordinates translation.
        '''
        if isinstance(other, BaseCoordinateFrame):
            other = other.transform_to(self)._data
        return self.realize_frame(other + self._data)


    def __sub__(self, other: Union[BaseRepresentation, BaseCoordinateFrame])   \
        -> ExtendedCoordinateFrame:
        '''Left hand side coordinates subtraction.
        '''
        if isinstance(other, BaseCoordinateFrame):
            other = other.transform_to(self)._data
        return self.realize_frame(self._data - other)


    def __rsub__(self, other: Union[BaseRepresentation, BaseCoordinateFrame])  \
        -> ExtendedCoordinateFrame:
        '''Right hand side coordinates subtraction.
        '''
        if isinstance(other, BaseCoordinateFrame):
            other = other.transform_to(self)._data
        return self.realize_frame(other - self._data)


    def __mul__(self, other: Any) -> ExtendedCoordinateFrame:
        '''Left hand side coordinates multiplication.
        '''
        return self.realize_frame(self._data * other)


    def __rmul__(self, other: Any) -> ExtendedCoordinateFrame:
        '''Right hand side coordinates multiplication.
        '''
        return self.realize_frame(other * self._data)


    def represent_as(self, base: BaseRepresentation, s: str='base',
        in_frame_units: bool=False) -> BaseRepresentation:

        r = super().represent_as(base, s, in_frame_units)
        self._protect(r)

        return r


    def _protect(self, representation: BaseRepresentation) -> None:
        for component in representation.components:
            getattr(representation, component).flags.writeable = False


    def _replicate(self, data: Any, copy: bool=False, **kwargs: Any) -> ECEF:
        if kwargs:
            return super()._replicate(data, copy, **kwargs)

        # Copy the object instead of calling the initialization again
        tmp = self.data, self.cache # type: ignore
        frame = _copy.copy(self) if copy is False else _copy.deepcopy(self)
        frame.cache = defaultdict(dict) # type: ignore
        if data is not None:
            if copy:
                data = data.copy()
            self._protect(data)
        frame._data = data
        self.data, self.cache = tmp

        return frame


    @property
    def data(self) -> BaseRepresentation:
        '''The coordinate data for this object.
        '''
        return self._data

    @data.setter
    def data(self, value: Optional[BaseRepresentation]) -> None:
        if value is not None:
            self._protect(value)
        self._data = value
        self.cache.clear()


class ECEF(ExtendedCoordinateFrame):
    '''Earth-Centered Earth-Fixed frame, co-moving with the Earth.
    '''

    default_representation: Final = CartesianRepresentation
    '''Default representation of local frames.'''

    obstime = TimeAttribute(default=None)
    '''The observation time. Defaults to `None`.'''


    def __init__(self, *args, obstime: Union[datetime, Time, str, None]=None,
                 **kwargs) -> None:
        super().__init__(*args, obstime=obstime, **kwargs)


    @property
    def earth_location(self) -> EarthLocation:
        '''The data in this frame as an
        :py:class:`~astropy.coordinates.EarthLocation`.
        '''
        geo = self.represent_as(GeodeticRepresentation)
        return EarthLocation.from_geodetic(geo.longitude.copy(), geo.latitude,
                                           geo.height, ellipsoid='WGS84')


@frame_transform_graph.transform(FunctionTransform, ITRS, ECEF)
def itrs_to_ecef(itrs: ITRS, ecef: ECEF) -> ECEF:
    '''Compute the transformation from ITRS to ECEF coordinates.
    '''
    if ecef._obstime == None:
        # The conversion goes to a generic frame
        ecef._obstime = itrs._obstime
    elif ecef._obstime != itrs._obstime:
        itrs = itrs.transform_to(ITRS(obstime=ecef._obstime))

    return ecef.realize_frame(itrs.cartesian)


@frame_transform_graph.transform(FunctionTransform, ECEF, ITRS)
def ecef_to_itrs(ecef: ECEF, itrs: ITRS) -> ITRS:
    '''Compute the transformation from ECEF to ITRS coordinates.
    '''
    if itrs._obstime == ecef._obstime:
        c = ecef.cartesian
    else:
        itrs0 = ITRS(ecef.cartesian, obstime=ecef._obstime)
        c = itrs0.transform_to(ITRS(obstime=itrs._obstime)).cartesian

    return itrs.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, ECEF, ECEF)
def ecef_to_ecef(ecef0: ECEF, ecef1: ECEF) -> ECEF:
    '''Compute the transformation from ECEF to ECEF coordinate.
    '''
    if ecef1._obstime is None:
        ecef1._obstime = ecef0._obstime

    return ecef1.realize_frame(ecef0.cartesian)


class LTP(ExtendedCoordinateFrame):
    '''Local frame tangent to the WGS84 ellipsoid and oriented along cardinal
       directions.
    '''

    default_representation: Final = CartesianRepresentation
    '''Default representation of local frames.'''

    location = Attribute(default=None)
    '''The origin on Earth of the local frame.'''

    orientation = Attribute(default='NWU')
    '''The cardinal directions of the x, y, and z axis (default: ENU).'''

    magnetic = Attribute(default=True)
    '''Use the magnetic north instead of the geographic one (default: True).'''

    declination = Attribute(default=None)
    '''Use the magnetic north with the given declination (default: None).'''

    rotation = Attribute(default=None)
    '''An optional rotation w.r.t. the cardinal directions.'''

    obstime = TimeAttribute(default=None)
    '''The observation time.'''


    def __init__(self, *args,
                 location: Union['EarthLocation', 'ECEF', 'LTP']=None,
                 orientation: Sequence[str]=None,
                 magnetic: Optional[bool]=None,
                 declination: Optional[u.Quantity]=None,
                 rotation: Optional[Rotation]=None,
                 obstime: Union['datetime', 'Time', str, None]=None,
                 **kwargs) -> None:

        # Do the base initialisation
        location = self.location if location is None else location
        try:
            location = location.earth_location
        except AttributeError:
            pass
        orientation = self.orientation if orientation is None else orientation

        super().__init__(*args, location=location, orientation=orientation,
                         magnetic=magnetic, declination=declination,
                         rotation=rotation, obstime=obstime, **kwargs)

        # Set the transform parameters
        itrs = self._location.itrs
        geo = itrs.represent_as(GeodeticRepresentation)
        latitude, longitude = geo.latitude / u.deg, geo.longitude / u.deg

        if magnetic and declination is None:
            # Compute the magnetic declination
            ecef = ECEF(itrs.x, itrs.y, itrs.z, obstime=self._obstime)

            if not _HAS_GEOMAGNET:
                from ..geomagnet import field as _geomagnetic_field
            field = _geomagnetic_field(ecef)

            c = field.cartesian
            c /= c.norm()
            h = c.represent_as(HorizontalRepresentation)
            declination = h.azimuth

        if declination is None:
            azimuth0 = 0
        else:
            azimuth0 = declination.to_value(u.deg)

        def vector(name):
            tag = name[0].upper()
            if tag == 'E':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   90 + azimuth0, 0)
            elif tag == 'W':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   270 + azimuth0, 0)
            elif tag == 'N':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   azimuth0,  0)
            elif tag == 'S':
                return turtle.ecef_from_horizontal(latitude, longitude,
                                                   180 + azimuth0,  0)
            elif tag == 'U':
                return turtle.ecef_from_horizontal(latitude, longitude, 0, 90)
            elif tag == 'D':
                return turtle.ecef_from_horizontal(latitude, longitude, 0, -90)
            else:
                raise ValueError(f'Invalid frame orientation `{name}`')

        ux = vector(self._orientation[0])
        uy = vector(self._orientation[1])
        uz = vector(self._orientation[2])

        self._basis = numpy.column_stack((ux, uy, uz))
        self._origin = itrs.cartesian

        if rotation is not None:
            self._basis = rotation.apply(self._basis, inverse=True)


    def rotated(self, rotation: Rotation, copy: bool=True) -> LTP:
        '''Get a rotated version of this frame.
        '''
        r = rotation if self.rotation is None else rotation * self.rotation
        frame = self._replicate(self.data, copy)
        frame._rotation = r
        frame._basis = rotation.apply(frame._basis, inverse=True)
        return frame


    @property
    def earth_location(self) -> EarthLocation:
        '''The data in this frame as an
        :class:`~astropy.coordinates.EarthLocation`.
        '''
        geo = self.transform_to(ECEF).represent_as(GeodeticRepresentation)
        return EarthLocation(lon=geo.longitude, lat=geo.latitude,
                             height=geo.height)


@frame_transform_graph.transform(FunctionTransform, ECEF, LTP)
def ecef_to_ltp(ecef: ECEF, ltp: LTP) -> LTP:
    '''Compute the transformation from ECEF to LTP coordinates.
    '''
    c = ecef.cartesian
    p = c.get_xyz()
    if c.x.unit.is_equivalent('m'):
        t = ltp._origin
        p[0] -= t.x
        p[1] -= t.y
        p[2] -= t.z
    p[:] = numpy.dot(ltp._basis.T, p)
    c = CartesianRepresentation(p, copy=False)

    if ltp._obstime is None:
        ltp._obstime = ecef._obstime

    return ltp.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, LTP, ECEF)
def ltp_to_ecef(ltp: LTP, ecef: ECEF) -> ECEF:
    '''Compute the transformation from LTP to ECEF coordinates.
    '''
    c = ltp.cartesian
    p = numpy.dot(ltp._basis, c.get_xyz())
    if c.x.unit.is_equivalent('m'):
        t = ltp._origin
        p[0] += t.x
        p[1] += t.y
        p[2] += t.z
    c = CartesianRepresentation(p, copy=False)

    if ecef._obstime is None:
        ecef._obstime = ltp._obstime

    return ecef.realize_frame(c)


@frame_transform_graph.transform(FunctionTransform, LTP, LTP)
def ltp_to_ltp(ltp0: LTP, ltp1: LTP) -> LTP:
    '''Compute the transformation from LTP to LTP coordinates.
    '''
    c = ltp0.cartesian
    translate = c.x.unit.is_equivalent('m')

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

    # Transform to and back from ECEF
    p = numpy.dot(ltp0._basis, c.get_xyz())
    if translate:
        t = ltp0._origin - ltp1._origin
        p[0] += t.x
        p[1] += t.y
        p[2] += t.z
    p[:] = numpy.dot(ltp1._basis.T, p)
    c = CartesianRepresentation(p, copy=False)

    return ltp1.realize_frame(c)
