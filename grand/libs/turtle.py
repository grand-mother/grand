'''Wrapper for the TURTLE C library
'''

from __future__ import annotations

from pathlib import Path
from typing import cast, Optional, Union
import weakref

from .._core import ffi, lib
import numpy


__all__ = ['LibraryError', 'Map', 'Stack', 'Stepper', 'ecef_from_geodetic',
           'ecef_from_horizontal', 'ecef_to_geodetic', 'ecef_to_horizontal']


class LibraryError(RuntimeError):
    '''A TURTLE library error'''

    def __init__(self, code: int):
        '''Set a TURTLE library error

        Parameters
        ----------
        code : int
            The function return code
        '''
        self.code = code
        message = ffi.string(lib.grand_error_get())

        header = 'A TURTLE library error occurred'
        if message is not None:
            message = f'{header}: {message}'
        else:
            message = header

        super().__init__(message)


def _regularize(a):
    '''Regularize an array (or float) input'''
    a = numpy.asanyarray(a)
    return numpy.require(a, float, ['CONTIGUOUS', 'ALIGNED'])


def ecef_from_geodetic(latitude, longitude, altitude):
    '''Convert geodetic coordinates to ECEF ones'''

    latitude, longitude, altitude = map(_regularize, (latitude, longitude,
                                                      altitude))
    if latitude.size != longitude.size:
        raise ValueError('latitude and longitude must have the same size')
    if latitude.size != altitude.size:
        raise ValueError('latitude and altitude must have the same size')

    if latitude.size == 1:
        ecef = numpy.zeros(3)
    else:
        ecef = numpy.zeros((latitude.size, 3))

    lib.turtle_ecef_from_geodetic_v(
        ffi.cast('double *', latitude.ctypes.data),
        ffi.cast('double *', longitude.ctypes.data),
        ffi.cast('double *', altitude.ctypes.data),
        ffi.cast('double *', ecef.ctypes.data),
        latitude.size
    )
    return ecef


def ecef_from_horizontal(latitude, longitude, azimuth, elevation):
    '''Convert horizontal coordinates to an ECEF direction'''

    latitude, longitude, azimuth, elevation = map(_regularize,
        (latitude, longitude, azimuth, elevation))
    if latitude.size != longitude.size:
        raise ValueError('latitude and longitude must have the same size')
    if latitude.size != azimuth.size:
        raise ValueError('latitude and azimuth must have the same size')
    if latitude.size != elevation.size:
        raise ValueError('latitude and elevation must have the same size')

    if latitude.size == 1:
        direction = numpy.zeros(3)
    else:
        direction = numpy.zeros((latitude.size, 3))

    lib.turtle_ecef_from_horizontal_v(
        ffi.cast('double *', latitude.ctypes.data),
        ffi.cast('double *', longitude.ctypes.data),
        ffi.cast('double *', azimuth.ctypes.data),
        ffi.cast('double *', elevation.ctypes.data),
        ffi.cast('double *', direction.ctypes.data),
        latitude.size
    )
    return direction


def ecef_to_geodetic(ecef):
    '''Convert ECEF coordinates to geodetic ones'''

    ecef = _regularize(ecef)
    if (ecef.size < 3) or ((ecef.size % 3) != 0):
        raise ValueError('ecef coordinates must be n x 3')

    n = int(ecef.size / 3)
    if n == 1:
        set0 = lambda x:numpy.zeros(1)
        latitude, longitude, altitude = map(set0, range(3)) 
    else:
        set0 = lambda x: numpy.zeros(n)
        latitude, longitude, altitude = map(set0, range(3))

    lib.turtle_ecef_to_geodetic_v(
        ffi.cast('double *', ecef.ctypes.data),
        ffi.cast('double *', latitude.ctypes.data),
        ffi.cast('double *', longitude.ctypes.data),
        ffi.cast('double *', altitude.ctypes.data),
        latitude.size)

    if n == 1:
        return latitude[0], longitude[0], altitude[0]
    else:
        return latitude, longitude, altitude


def ecef_to_horizontal(latitude, longitude, direction):
    '''Convert an ECEF direction to horizontal coordinates'''

    latitude, longitude, direction = map(_regularize, (latitude, longitude,
                                                       direction))
    if latitude.size != longitude.size:
        raise ValueError('latitude and longitude must have the same size')
    if (direction.size < 3) or ((direction.size % 3) != 0):
        raise ValueError('direction must be n x 3')
    n = int(direction.size / 3)
    if latitude.size != n:
        raise ValueError('latitude and direction must have consistent shapes')

    if n == 1:
        set0 = lambda x:numpy.zeros(1)
        azimuth, elevation = map(set0, range(2))
    else:
        set0 = lambda x: numpy.zeros(n)
        azimuth, elevation = map(set0, range(2))

    lib.turtle_ecef_to_horizontal_v(
        ffi.cast('double *', latitude.ctypes.data),
        ffi.cast('double *', longitude.ctypes.data),
        ffi.cast('double *', direction.ctypes.data),
        ffi.cast('double *', azimuth.ctypes.data),
        ffi.cast('double *', elevation.ctypes.data),
        latitude.size)

    if n == 1:
        return azimuth[0], elevation[0] if n == 1 else azimuth, elevation
    else:
        return azimuth, elevation


class Map:
    '''Proxy for a TURTLE map object'''

    def __init__(self, path: Union[Path, str]):
        '''Initialise a map object from a data file

        Parameters
        ----------
        path : Path or str
            The path where the data are located

        Raises
        ------
        LibraryError
            A TURTLE library error occured, e.g. if the data could not be loaded
        '''

        # Create the map object
        map_ = ffi.new('struct turtle_map **')
        path_ = ffi.new('char []', str(path).encode())

        r = lib.turtle_map_load(map_, path_)
        if r != 0:
            self._map, self._path = None, None
            raise LibraryError(r)
        else:
            self._map = map_
            self._path = path


        def destroy():
            lib.turtle_map_destroy(self._map)
            self._map = None

        weakref.finalize(self, destroy)


    def elevation(self, x, y):
        '''Get the elevation at the given map coordinates'''

        x, y = map(_regularize, (x, y))
        if x.size != y.size:
            raise ValueError('x and y must have the same size')

        n = x.size
        elevation = numpy.zeros(n)

        if self._map is None:
            elevation = numpy.nan
            return elevation
        else:
            lib.turtle_map_elevation_v(self._map[0],
                ffi.cast('double *', x.ctypes.data),
                ffi.cast('double *', y.ctypes.data),
                ffi.cast('double *', elevation.ctypes.data), n)
            return elevation[0] if n == 1 else elevation


    @property
    def path(self):
        '''The path where the data tiles are located'''
        return self._path


class Stack:
    '''Proxy for a TURTLE stack object'''

    def __init__(self, path: Union[Path, str], stack_size: int=0):
        '''Create a stack of maps for a world wide topography model

        Parameters
        ----------
        path : Path or str
            The path where the data tiles are located
        stack_size : integer, optional
            The maximum number of data tiles kept in memory

        Raises
        ------
        LibraryError
            A TURTLE library error occured, e.g. if the data format is not valid
        '''
        self._stack, self._path, self._stack_size = None, None, None

        # Create the stack object
        stack_ = ffi.new('struct turtle_stack **')
        path_ = ffi.new('char []', str(path).encode())

        r = lib.turtle_stack_create(stack_, path_, stack_size,
                                    ffi.NULL, ffi.NULL)
        if r != 0: raise LibraryError(r)
        self._stack = stack_
        self._path = path
        self._stack_size = stack_size


        def destroy():
            lib.turtle_stack_destroy(self._stack)
            self._stack = None

        weakref.finalize(self, destroy)


    def elevation(self, latitude, longitude):
        '''Get the elevation at the given geodetic coordinates'''

        latitude, longitude = map(_regularize, (latitude, longitude))
        if latitude.size != longitude.size:
            raise ValueError('latitude and longitude must have the same size')

        n = latitude.size
        elevation = numpy.zeros(n)

        lib.turtle_stack_elevation_v(self._stack[0],
            ffi.cast('double *', latitude.ctypes.data),
            ffi.cast('double *', longitude.ctypes.data),
            ffi.cast('double *', elevation.ctypes.data), n)
        return elevation[0] if n == 1 else elevation


    @property
    def path(self):
        '''The path where the data tiles are located'''
        return self._path


    @property
    def stack_size(self):
        '''The maximum number of data tiles kept in memory'''
        return self._stack_size


class Stepper:
    '''Proxy for a TURTLE stepper object'''

    def __init__(self):
        '''Create a stepper for the ray tracing of topography data
        '''
        stepper_ = ffi.new('struct turtle_stepper **')

        r = lib.turtle_stepper_create(stepper_)
        if r != 0:
            self._stepper = None
            raise LibraryError(r)
        self._stepper = stepper_
        self._geoid = None
        self._data = set([])

        def destroy():
            lib.turtle_stepper_destroy(self._stepper)
            self._stepper = None

        weakref.finalize(self, destroy)


    def add(self, data: Union[Map, Stack, None]=None, offset: float=0):
        if data is not None:
            if isinstance(data, Map):
                if data._map is None:
                    raise ValueError('no data')
                lib.turtle_stepper_add_map(self._stepper[0], data._map[0],
                                           offset)
            else:
                if data._stack is None:
                    raise ValueError('no data')
                lib.turtle_stepper_add_stack(self._stepper[0], data._stack[0],
                                             offset)
            self._data.add(data)
        else:
            lib.turtle_stepper_add_flat(self._stepper[0], offset)


    @property
    def geoid(self):
        return self._geoid

    @geoid.setter
    def geoid(self, map_: Optional[Map]):
        if map_ is None:
            self._data.pop(self._geoid)
            self._geoid = None
        else:
            self._data.add(map_)
            self._geoid = map_
            if map_._map is None:
                raise ValueError('no data')
            map_ = map_._map[0]
        lib.turtle_stepper_geoid_set(self._stepper[0], map_)

