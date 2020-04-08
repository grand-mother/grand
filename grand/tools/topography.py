'''Topography wrapper for GRAND packages.
'''

from __future__ import annotations

import enum
import os
from pathlib import Path
from typing import Optional, Union
from typing_extensions import Final

import astropy.units as u
import numpy

from . import DATADIR
from .coordinates import ECEF, GeodeticRepresentation, LTP
from ..libs.turtle import Map as _Map, Stack as _Stack, Stepper as _Stepper
from .. import store
from .._core import ffi, lib

__all__ = ['elevation', 'distance', 'geoid_undulation', 'update_data',
           'cachedir', 'model', 'Reference', 'Topography']


_CACHEDIR: Final = Path(__file__).parent / 'data' / 'topography'
'''Location of cached topography data'''


_DEFAULT_MODEL: Final = 'SRTMGL1'
'''The default topographic model'''


_default_topography: Optional['Topography'] = None
'''Stack for the topographic data'''


_geoid: Optional[_Map] = None
'''Map with geoid undulations'''


class Reference(enum.IntEnum):
    '''Reference level for topography data
    '''

    ELLIPSOID = enum.auto()
    GEOID = enum.auto()
    LOCAL = enum.auto()


def distance(position: Union[ECEF, LTP], direction: Union[ECEF, LTP],
    maximum_distance: Optional[u.Quantity]=None) -> u.Quantity:
    '''Get the signed intersection distance with the topography.
    '''
    global _default_topography

    if _default_topography is None:
        _CACHEDIR.mkdir(exist_ok=True)
        _default_topography = Topography(_CACHEDIR)
    return _default_topography.distance(position, direction, maximum_distance)


def elevation(coordinates: Union[ECEF, LTP],
    reference: Optional[Reference]=None) -> u.Quantity:
    '''Get the topography elevation, w.r.t. sea level or w.r.t. the ellipsoid.
    '''
    global _default_topography

    if _default_topography is None:
        _CACHEDIR.mkdir(exist_ok=True)
        _default_topography = Topography(_CACHEDIR)
    return _default_topography.elevation(coordinates, reference)


def _get_geoid():
    global _geoid

    if _geoid is None:
        path = os.path.join(DATADIR, 'egm96.png')
        _geoid = _Map(path)
    return _geoid


def geoid_undulation(coordinates: Union[ECEF, LTP]) -> u.Quantity:
    '''Get the geoid undulation.
    '''
    geoid = _get_geoid()

    # Compute the geodetic coordinates
    geodetic = coordinates.transform_to(ECEF).represent_as(
               GeodeticRepresentation)

    z = geoid.elevation(geodetic.longitude.to_value(u.deg),
                         geodetic.latitude.to_value(u.deg))
    return z << u.m


def update_data(coordinates: Union[ECEF, LTP]=None, clear: bool=False,
                radius: u.Quantity=None):
    '''Update the cache of topography data.
    '''
    if clear:
        for p in _CACHEDIR.glob('**/*.*'):
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
                                          longitude[i] * u.deg)),
                        orientation='ENU', magnetic=False)
                c = c.transform_to(ECEF).represent_as(GeodeticRepresentation)
                latitude[i] = c.latitude / u.deg
                longitude[i] = c.longitude / u.deg


        # Get the corresponding tiles
        longitude = [int(numpy.floor(lon)) for lon in longitude]
        latitude = [int(numpy.floor(lat)) for lat in latitude]

        for lat in range(latitude[0], latitude[1] + 1):
            for lon in range(longitude[0], longitude[1] + 1):
                if lat < 0:
                    ns = 'S'
                    lat = -lat
                else:
                    ns = 'N'

                if lon < 0:
                    ew = 'W'
                    lon = -lon
                else:
                    ew = 'E'

                basename = f'{ns}{lat:02.0f}{ew}{lon:03.0f}.SRTMGL1.hgt'
                path = _CACHEDIR / basename
                if not path.exists():
                    try:
                        data = store.get(basename)
                    except store.InvalidBLOB:
                        raise ValueError(f'missing data for {basename}')   \
                        from None
                    else:
                        with path.open('wb') as f:
                            f.write(data)

    # Reset the topography proxy
    global _default_topography
    _default_topography = None


def cachedir() -> Path:
    '''Get the location of the topography data cache.
    '''
    return _CACHEDIR


def model() -> str:
    '''Get the default model for topographic data.
    '''
    return _DEFAULT_MODEL


class Topography:
    '''Proxy to topography data.
    '''

    def __init__(self, path: Union[Path, str]) -> None:
        self._stack = _Stack(str(path))
        self._stepper:Optional[_Stepper] = None


    def elevation(self, coordinates: Union[ECEF, LTP],
        reference: Optional[Reference]=None) -> u.Quantity:
        '''Get the topography elevation, w.r.t. sea level, w.r.t the
           ellipsoid or in local coordinates.
        '''

        if reference is None:
            if isinstance(coordinates, LTP):
                elevation = self._local_elevation(coordinates)
            else:
                geoid = _get_geoid()._map[0]
                elevation = self._global_elevation(coordinates,
                                                   Reference.ELLIPSOID)
        else:
            if reference == Reference.LOCAL:
                if not isinstance(coordinates, LTP):
                    raise ValueError('not an LTP frame')
                elevation = self._local_elevation(coordinates)
            else:
                elevation = self._global_elevation(coordinates, reference)

        if elevation.size == 1:
            elevation = elevation[0]

        return elevation << u.m


    @staticmethod
    def _as_double_ptr(a):
        a = numpy.require(a, float, ['CONTIGUOUS', 'ALIGNED'])
        return ffi.cast('double *', a.ctypes.data)


    def _local_elevation(self, coordinates: LTP) -> u.Quantity:
        '''Get the topography elevation in local coordinates, i.e. along the
           (Oz) axis.
        '''

        # Compute the geodetic coordinates
        cartesian = coordinates.cartesian
        x = cartesian.x.to_value(u.m)
        y = cartesian.y.to_value(u.m)
        if not isinstance(x, numpy.ndarray):
            x = numpy.array((x,))
            y = numpy.array((y,))

        # Return the topography elevation
        n = x.size
        elevation = numpy.zeros(n)
        origin = coordinates._origin.xyz.to_value(u.m)
        geoid = _get_geoid()._map[0]
        stack = self._stack._stack[0] if self._stack._stack else ffi.NULL

        lib.grand_topography_local_elevation(stack, geoid,
            self._as_double_ptr(origin),
            self._as_double_ptr(coordinates._basis),
            self._as_double_ptr(x),
            self._as_double_ptr(y),
            self._as_double_ptr(elevation), n)

        return elevation


    def _global_elevation(self, coordinates: Union[ECEF, LTP],
        reference: Reference) -> u.Quantity:
        '''Get the topography elevation w.r.t. sea level or w.r.t. the
           ellipsoid.
        '''

        # Compute the geodetic coordinates
        geodetic = coordinates.transform_to(ECEF).represent_as(
                   GeodeticRepresentation)
        latitude = geodetic.latitude.to_value('deg')
        longitude = geodetic.longitude.to_value('deg')
        if not isinstance(latitude, numpy.ndarray):
            latitude = numpy.array((latitude,))
            longitude = numpy.array((longitude,))

        # Return the topography elevation
        n = latitude.size
        elevation = numpy.zeros(n)
        if reference == Reference.ELLIPSOID:
            geoid = _get_geoid()._map[0]
        else:
            geoid = ffi.NULL
        stack = self._stack._stack[0] if self._stack._stack else ffi.NULL

        lib.grand_topography_global_elevation(stack, geoid,
            self._as_double_ptr(latitude),
            self._as_double_ptr(longitude),
            self._as_double_ptr(elevation), n)

        return elevation


    def distance(self, position: Union[ECEF, LTP],
                 direction: Union[ECEF, LTP],
                 maximum_distance: Optional[u.Quantity]=None) -> u.Quantity:
        '''Get the signed intersection distance with the topography.
        '''

        if self._stepper is None:
            stepper = _Stepper()
            stepper.add(self._stack)
            stepper.geoid = _get_geoid()
            self._stepper = stepper

        position = position.transform_to(ECEF).cartesian
        direction = direction.transform_to(ECEF).cartesian
        dn = maximum_distance.size if maximum_distance is not None else 1
        n = max(position.x.size, direction.x.size, dn)

        if ((direction.size > 1) and (direction.size < n)) or                  \
           ((position.size > 1) and (position.size < n)) or                    \
           ((dn > 1) and (dn < n)):
            raise ValueError('incompatible size')

        r = numpy.empty(3 * n)
        v = numpy.empty(3 * n)
        d = numpy.empty(n)

        r[::3] = position.x.to_value('m')
        r[1::3] = position.y.to_value('m')
        r[2::3] = position.z.to_value('m')
        v[::3] = direction.x.value
        v[1::3] = direction.y.value
        v[2::3] = direction.z.value
        d[:] = maximum_distance.to_value('m') if maximum_distance is not None  \
                                              else 0

        lib.grand_topography_distance(
            self._stepper._stepper[0],
            self._as_double_ptr(r),
            self._as_double_ptr(v),
            self._as_double_ptr(d), n)

        if d.size == 1: d = d[0]
        return d << u.m
