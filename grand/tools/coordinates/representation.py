'''Extra representations for astropy.coordinates
'''

from typing import Union
from typing_extensions import Final

from collections import OrderedDict

from ...libs import turtle

import numpy
import astropy.units as u
from astropy.coordinates import BaseRepresentation, CartesianRepresentation,   \
                                Latitude, Longitude

__all__ = ['GeodeticRepresentation', 'HorizontalRepresentation']


class GeodeticRepresentation(BaseRepresentation):
    '''Geodetic coordinates representation w.r.t. the WGS84 ellipsoid.'''

    attr_classes: Final = OrderedDict([('latitude', Latitude),
                                       ('longitude', Longitude),
                                       ('height', u.Quantity)])
    '''Attributes of a Geodetic representation.'''


    def __init__(self, latitude: Latitude, longitude: Longitude,
                 height: u.Quantity=0 * u.m, copy: bool=True) -> None:
        super().__init__(latitude, longitude, height, copy=copy)


    @classmethod
    def from_cartesian(cls, cart: CartesianRepresentation) ->                  \
            'GeodeticRepresentation':
        '''Generate a Geodetic representation from a Cartesian one.
        '''
        m1 = 1 / u.m
        x, y, z = map(lambda v: v * m1, (cart.x, cart.y, cart.z))
        if x.size > 1:
            ecef = numpy.column_stack((x, y, z))
        else:
            ecef = (x, y, z)

        geodetic = turtle.ecef_to_geodetic(ecef)
        return cls(geodetic[0] * u.deg, geodetic[1] * u.deg, geodetic[2] * u.m,
                   copy=False)


    def to_cartesian(self) -> CartesianRepresentation:
        '''Generate a Cartesian representation from a Geodetic one.
        '''
        d1, m1 = 1 / u.deg, 1 / u.m
        ecef = turtle.ecef_from_geodetic(self.latitude * d1,
                                         self.longitude * d1, self.height * m1)
        if ecef.size == 3:
            return CartesianRepresentation(ecef[0] * u.m, ecef[1] * u.m,
                                           ecef[2] * u.m, copy=False)
        else:
            return CartesianRepresentation(ecef[:,0] * u.m, ecef[:,1] * u.m,
                                           ecef[:,2] * u.m, copy=False)


class HorizontalRepresentation(BaseRepresentation):
    '''Horizontal angular representation, for unit vectors.'''


    attr_classes: Final = OrderedDict([('azimuth', Longitude),
                                       ('elevation', Latitude)])
    '''Attributes of a Horizontal representation'''


    def __init__(self, azimuth: Union[Longitude, u.Quantity, str],
                 elevation: Union[Latitude, u.Quantity, str],
                 copy: bool=True) -> None:
        azimuth = Longitude(azimuth)
        azimuth.wrap_angle = 180 * u.deg

        super().__init__(azimuth, elevation, copy=copy)


    @classmethod
    def from_cartesian(cls, cart: CartesianRepresentation) ->                  \
            'HorizontalRepresentation':
        '''Generate a Horizontal angular representation from a Cartesian unit
        vector.
        '''
        if ((cart.x.unit != u.one) or (cart.y.unit != u.one) or
            (cart.z.unit != u.one)):
            raise ValueError('coordinates must be dimensionless')

        rho = numpy.sqrt(cart.x**2 + cart.y**2)
        theta = numpy.arctan2(rho, cart.z)

        if theta == 0 * u.rad:
            elevation = 90 * u.deg
            azimuth = 0 * u.deg
        else:
            elevation = 90 * u.deg - theta
            azimuth = 90 * u.deg - numpy.arctan2(cart.y, cart.x)

        return cls(azimuth, elevation, copy=False)


    def to_cartesian(self) -> CartesianRepresentation:
        '''Generate a Cartesian unit vector from this Horizontal angular
        representation.
        '''
        theta = 90 * u.deg - self.elevation
        phi = 90 * u.deg - self.azimuth
        ct, st = numpy.cos(theta), numpy.sin(theta)

        return CartesianRepresentation(numpy.cos(phi) * st,
                                       numpy.sin(phi) * st, ct, copy=False)
