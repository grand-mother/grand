"""
Fundamental Coordinate Data Structure:
	np.array([[x1, x2, ..], 
			  [y1, y2, ..], 
			  [z1, z2, ..]]
	np.array([[x1], 
			  [y1], 
			  [z1]]

units: 
	energy: GeV
	time  : nanosecond
	length: m
	voltage: microVolt
	Efield: microVolt/m
	Bfield: nanoTesla
	angles: degree
	grammage: g/cm2
	Density: g/cm3 
	frequency: Hz
	resistance: Ohm

"""


from __future__ import annotations
from typing import Optional, Tuple, Union, Any
from datetime import datetime
import copy as _copy
import enum
import os
from numbers import Number
from logging import getLogger

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None

from ..libs import turtle
from . import DATADIR

logger = getLogger(__name__)

# Mean value of proposed GP300 layout. Just a placeholder for the default GP300 origin.
grd_origin_lat = 38.88849  # degree
grd_origin_lon = 92.28605  # degree
grd_origin_height = 2920.522  # meter

__all__ = (
    "Coordinates",
    "CartesianRepresentation",
    "SphericalRepresentation",
    "HorizontalRepresentation",
    "Horizontal",
    "HorizontalVector",
    "ECEF",
    "Geodetic",
    "LTP",
    "GRANDCS",
    "_cartesian_to_spherical",
    "_cartesian_to_horizontal",
    "_spherical_to_cartesian",
    "_spherical_to_horizontal",
    "_horizontal_to_cartesian",
    "_horizontal_to_spherical",
    "Reference",
    "geoid_undulation",
)


def copy(obj, deep=False, attributes=[]):
    if deep:
        new = _copy.deepcopy(obj)
        for var in vars(obj):
            setattr(new, var, _copy.deepcopy(getattr(obj, var)))
    else:
        new = _copy.copy(obj)
        for var in vars(obj):
            setattr(new, var, _copy.copy(getattr(obj, var)))

    return new


# Using Reference and geoid_undulation from topography.py gives circular import error.
# So simple Reference and geoid_undulation functions are defined here for coordinates purpose only.
class Reference(enum.IntEnum):
    """Reference level for height in Geodetic coordinate system.
    Reference is also defined in topography.py for Topography use.
    """

    ELLIPSOID = enum.auto()
    GEOID = enum.auto()


def geoid_undulation(latitude=None, longitude=None):
    """Get the geoid undulation. This function calculates the height of
    the geoid w.r.t the ellipsoid at a given latitude and longitude.
    This function is also defined in topography.py for Topography use.
    """
    path = os.path.join(DATADIR, "egm96.png")
    geoid = turtle.Map(path)
    logger.info(f"geoid_undulation for {latitude} {longitude}")
    return geoid.elevation(longitude, latitude)


# Define functions to transform from one coordinate representation to
# another coordinate representation. Cartesian, Spherical, and Horizontal
# coordinate representation are defined.
def _cartesian_to_spherical(
    x: Union[float, int, np.ndarray],
    y: Union[float, int, np.ndarray],
    z: Union[float, int, np.ndarray],
) -> Union[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    """Transform Cartesian coordinates to spherical"""
    rho2 = x**2 + y**2
    rho = np.sqrt(rho2)
    theta = np.rad2deg(np.arctan2(rho, z))
    phi = np.rad2deg(np.arctan2(y, x))
    r = np.sqrt(rho2 + z**2)

    return theta, phi, r


# Horizontal has an axis fixed to geographic North, so it can not be
# converted like cartesian and spherical. If conversion is done, then
# the same origin for both and ENU basis for cartesian is assumed.
def _cartesian_to_horizontal(
    x: Union[float, int, np.ndarray],
    y: Union[float, int, np.ndarray],
    z: Union[float, int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Transform Cartesian coordinates to horizontal"""
    theta, phi, r = _cartesian_to_spherical(x, y, z)
    return _spherical_to_horizontal(theta, phi, r)


def _spherical_to_cartesian(
    theta: Union[float, int, np.ndarray],
    phi: Union[float, int, np.ndarray],
    r: Union[float, int, np.ndarray],
) -> Union[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    """Transform spherical coordinates to Cartesian"""
    cos_theta = np.cos(np.deg2rad(theta))
    sin_theta = np.sin(np.deg2rad(theta))

    x = r * np.cos(np.deg2rad(phi)) * sin_theta
    y = r * np.sin(np.deg2rad(phi)) * sin_theta
    z = r * cos_theta

    return x, y, z


def _spherical_to_horizontal(
    theta: Union[float, int, np.ndarray],
    phi: Union[float, int, np.ndarray],
    r: Union[float, int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Transform spherical coordinates to horizontal"""
    # return 0.5 * np.pi - phi, 0.5 * np.pi - theta, r
    return 90.0 - phi, 90.0 - theta, r


# Horizontal has an axis fixed to geographic North, so it can not be
# converted like cartesian and spherical. If conversion is done, then
# the same origin for both and ENU basis for cartesian is assumed.
def _horizontal_to_cartesian(
    azimuth: Union[float, int, np.ndarray],
    elevation: Union[float, int, np.ndarray],
    norm: Union[float, int, np.ndarray],
) -> Union[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    """Transform horizontal coordinates to Cartesian"""
    theta, phi, r = _horizontal_to_spherical(azimuth, elevation, norm)
    return _spherical_to_cartesian(theta, phi, r)


def _horizontal_to_spherical(
    azimuth: Union[float, int, np.ndarray],
    elevation: Union[float, int, np.ndarray],
    norm: Union[float, int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Transform horizontal coordinates to spherical"""
    # return 0.5 * np.pi - elevation, 0.5 * np.pi - azimuth, norm
    return 90.0 - elevation, 90.0 - azimuth, norm


# -----------Base Representation------------
class Coordinates(np.ndarray):
    """
    Generic container for a coordinates object
    This object created is a standard np.ndarray of size (3, n)
    where 3 is for 3D coordinates and n is the number of entries.
    """

    def __new__(cls, n: Optional[int] = None):
        """
        Create 3xn ndarray coordinates instance with n random entries for all 3D coordinate system.
        n: number of coordinate points.
        """
        if isinstance(n, int):
            return super().__new__(cls, (3, n), dtype="f8")
        else:
            raise TypeError(
                "Input number of coordinates point is type",
                type(n),
                "Integer is required.",
            )


# --------------Representation---------------
class CartesianRepresentation(Coordinates):
    """
    Generic container for cartesian coordinates
    """

    def __new__(
        cls,
        arg: Any = None,
        x: Union[float, np.ndarray] = None,
        y: Union[float, np.ndarray] = None,
        z: Union[float, np.ndarray] = None,
    ):
        """
        Create a Cartesian coordinates instance
        Unspecified coordinates are initialized with entry 0 in 3xn ndarray.
        n: number of coordinate points. 3xn np.ndarray object will be instantiated
           which will then be replaced by input x, y, and z. 'n' has to be predefined.
        """
        if isinstance(arg, SphericalRepresentation):
            x, y, z = _spherical_to_cartesian(arg.theta, arg.phi, arg.r)
        elif isinstance(arg, (CartesianRepresentation, ECEF, LTP, GRANDCS)):
            x, y, z = arg.x, arg.y, arg.z

        if isinstance(x, Number):
            n = 1
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray):
            n = len(x)
            assert n == len(y), (
                "Length of x and y array must be the same. \
				x: %i, y: %i"
                % (len(x), len(y))
            )
            assert n == len(z), (
                "Length of x and z array must be the same. \
							   x: %i, z: %i"
                % (len(x), len(z))
            )
        else:
            raise TypeError(type(x))

        # create 3xn ndarray coordinates instance with random entries.
        obj = super().__new__(cls, n)
        # replace x-coordinates with input x. x can be int, float, or ndarray.
        obj[0] = x
        # replace y-coordinates with input y. y can be int, float, or ndarray.
        obj[1] = y
        # replace z-coordinates with input z. z can be int, float, or ndarray.
        obj[2] = z
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, v):
        self[0] = v

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, v):
        self[1] = v

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, v):
        self[2] = v

    def cartesian_to_spherical(self):
        theta, phi, r = _cartesian_to_spherical(self[0], self[1], self[2])
        return SphericalRepresentation(theta=theta, phi=phi, r=r)

    def cartesian_to_horizontal(self):
        azi, ele, norm = _cartesian_to_horizontal(self[0], self[1], self[2])
        return HorizontalRepresentation(azimuth=azi, elevation=ele, norm=norm)

    def norm(self):
        return np.linalg.norm(self)


class SphericalRepresentation(Coordinates):
    """
    Generic container for spherical coordinates
    """

    def __new__(
        cls,
        arg: Any = None,
        theta: Union[float, int, np.ndarray] = None,
        phi: Union[float, int, np.ndarray] = None,
        r: Union[float, int, np.ndarray] = None,
    ):
        """
        Create a spherical coordinates instance.
        Object with (3,n) ndarray is created which will later be filled with input
        theta, phi, and r, where n is equal to len(theta). If predefined object
        (like CartesianCoordinates,..) is given as an argument, it will be
        first converted to spherical coordinates. Then (3,n) ndarray will be filled
        with converted theta, phi, and r.
        n: number of coordinate points. 3xn np.ndarray object will be instantiated
           which will then be replaced by input theta, phi, and r. 'n' has to be predefined.
        theta: angle from Z-axis towards XY plane. Also called zenith angle or colatitude. 0<=theta<=180 deg.
        phi  : angle from X-axis towards Y-axis in XY plane. 0<=phi<=360 deg.
        r    : magnitude of a vector or a distance to a point from the origin.
        """
        if isinstance(arg, CartesianRepresentation):
            theta, phi, r = _cartesian_to_spherical(arg.x, arg.y, arg.z)
        elif isinstance(arg, SphericalRepresentation):
            theta, phi, r = arg.theta, arg.phi, arg.r

        if isinstance(theta, Number):
            n = 1
        elif (
            isinstance(theta, np.ndarray)
            and isinstance(phi, np.ndarray)
            and isinstance(r, np.ndarray)
        ):
            n = len(theta)
            assert n == len(phi), (
                "Length of theta and phi array must be the same. \
								 theta: %i, phi: %i"
                % (n, len(phi))
            )
            assert n == len(r), (
                "Length of theta and r array must be the same. \
							   theta: %i, r: %i"
                % (n, len(r))
            )
        else:
            raise TypeError(type(theta))

        # create 3xn ndarray coordinates instance with random entries.
        obj = super().__new__(cls, n)
        # replace 0-coordinates with input theta. theta can be int, float, or ndarray.
        obj[0] = theta
        # replace 1-coordinates with input phi. phi can be int, float, or ndarray.
        obj[1] = phi
        # replace 2-coordinates with input r. r can be int, float, or ndarray.
        obj[2] = r
        return obj

    @property
    def theta(self):
        return self[0]

    @theta.setter
    def theta(self, v):
        self[0] = v

    @property
    def phi(self):
        return self[1]

    @phi.setter
    def phi(self, v):
        self[1] = v

    @property
    def r(self):
        return self[2]

    @r.setter
    def r(self, v):
        self[2] = v

    def spherical_to_cartesian(self):
        x, y, z = _spherical_to_cartesian(self[0], self[1], self[2])
        return CartesianRepresentation(x=x, y=y, z=z)

    def spherical_to_horizontal(self):
        azi, ele, norm = _spherical_to_horizontal(self[0], self[1], self[2])
        return HorizontalRepresentation(azimuth=azi, elevation=ele, norm=norm)


class HorizontalRepresentation(Coordinates):
    """
    Generic container for horizontal coordinates.
    """

    def __new__(
        cls,
        azimuth: Union[float, int, np.ndarray] = None,
        elevation: Union[float, int, np.ndarray] = None,
        norm: Union[float, int, np.ndarray] = 1.0,
    ):
        """
        Create a horizontal coordinates instance.
        Object with (3,n) ndarray is created which will later be filled with input
        azimuth, elevation, and norm, where n is equal to len(azimuth). If predefined
        object (like CartesianCoordinates,..) is given as an argument, it will be
        first converted to horizontal coordinates. Then (3,n) ndarray will be filled
        with converted azimuth, elevation, and norm.
        n        : number of coordinate points. 3xn np.ndarray object will be instantiated
                           which will then be replaced by input azimuth, elevation, and norm.
                           'n' has to be predefined.
        azimuth  : angle from true North towards East.
        elevation: angle from horizontal plane (NE plane) towards zenith.
        norm     : distance from the origin to the point.
        """
        if isinstance(azimuth, Number):
            n = 1
        elif (
            isinstance(azimuth, np.ndarray)
            and isinstance(elevation, np.ndarray)
            and isinstance(norm, np.ndarray)
        ):
            n = len(azimuth)
            assert n == len(elevation), (
                "Length of azimuth and elevation array must be the same. \
									   azimuth: %i, elevation: %i"
                % (n, len(elevation))
            )
        else:
            raise TypeError(type(azimuth))
        # create 3xn ndarray coordinates instance with random entries.
        obj = super().__new__(cls, n)
        # replace 0-coordinates with input azimuth. azimuth can be int, float, or ndarray.
        obj[0] = azimuth
        # replace 1-coordinates with input elevation. elevation can be int, float, or ndarray.
        obj[1] = elevation
        # replace 2-coordinates with input norm. norm can be int, float, or ndarray.
        obj[2] = norm

        return obj

    @property
    def azimuth(self):
        return self[0]

    @azimuth.setter
    def azimuth(self, v):
        self[0] = v

    @property
    def elevation(self):
        return self[1]

    @elevation.setter
    def elevation(self, v):
        self[1] = v

    @property
    def norm(self):
        return self[2]

    @norm.setter
    def norm(self, v):
        self[2] = v

    def horizontal_to_cartesian(self):
        x, y, z = _horizontal_to_cartesian(self[0], self[1], self[2])
        return CartesianRepresentation(x=x, y=y, z=z)

    def horizontal_to_spherical(self):
        th, phi, r = _horizontal_to_spherical(self[0], self[1], self[2])
        return SphericalRepresentation(theta=th, phi=phi, r=r)


class GeodeticRepresentation(Coordinates):
    """
    Generic container for Geodetic coordinate system. Center of this frame
    is the center of Earth. Geodetic representation w.r.t. the WGS84 ellipsoid.

    Latitude:	Angle north and south of the equator. +ve in the northern hemisphere,
                            -ve in the southern hemisphere. Range: -90 deg (South Pole)
                            to +90 deg (North Pole). In equator, latitude = 0.
    Longitude:	Angle east and west of the Prime Meridian. The Prime Meridian
                            is a north-south line that passes through Greenwich, UK.
                            +ve to the east of the Prime Meridian, -ve to the west.
                            Range: -180 deg to +180 deg.
    Height:	Also called altitude or elevation, this represents the height above
                    the Earth ellipsoid, measured in meters. The Earth ellipsoid is a
                    mathematical surface defined by a semi-major axis and a semi-minor axis.
                    The most common values for these two parameters are defined by
                    the World Geodetic Standard 1984 (WGS-84). The WGS-84 ellipsoid is
                    intended to correspond to mean sea level. A Geodetic height of zero
                    therefore roughly corresponds to sea level, with positive values increasing
                    away from the Earth’s center. The theoretical range of height values is
                    from the center of the Earth (about -6,371km) to positive infinity.

    """

    def __new__(
        cls,
        latitude: Union[float, int, np.ndarray] = None,
        longitude: Union[float, int, np.ndarray] = None,
        height: Union[float, int, np.ndarray] = None,
    ):
        """
        Create a new instance from latitude, longitude, and height.
        """
        if isinstance(latitude, Number):
            n = 1
        elif (
            isinstance(latitude, np.ndarray)
            and isinstance(longitude, np.ndarray)
            and isinstance(height, np.ndarray)
        ):
            n = len(latitude)
            assert n == len(longitude)
            assert n == len(height)

        else:
            raise TypeError(type(latitude))

        # create 3xn ndarray coordinates instance with random entries.
        obj = super().__new__(cls, n)
        # replace 0-position with input latitude. latitude can be int, float, or ndarray.
        obj[0] = latitude
        # replace 1-position with input longitude. longitude can be int, float, or ndarray.
        obj[1] = longitude
        # replace 2-position with input height. height can be int, float, or ndarray.
        obj[2] = height

        return obj

    @property
    def latitude(self):
        return self[0]

    @latitude.setter
    def latitude(self, v):
        self[0] = v

    @property
    def longitude(self):
        return self[1]

    @longitude.setter
    def longitude(self, v):
        self[1] = v

    @property
    def height(self):
        return self[2]

    @height.setter
    def height(self, v):
        self[2] = v


# ------------------Frame---------------------
class Geodetic(GeodeticRepresentation):
    """
    Generic container for Geodetic coordinate system. Center of this frame
    is the center of Earth.

    Latitude:	Angle north and south of the equator. +ve in the northern hemisphere,
                            -ve in the southern hemisphere. Range: -90 deg (South Pole)
                            to +90 deg (North Pole). In equator, latitude = 0.
    Longitude:	Angle east and west of the Prime Meridian. The Prime Meridian
                            is a north-south line that passes through Greenwich, UK.
                            +ve to the east of the Prime Meridian, -ve to the west.
                            Range: -180 deg to +180 deg.
    Height:	Also called altitude or elevation, this represents the height above
                    the Earth ellipsoid, measured in meters. The Earth ellipsoid is a
                    mathematical surface defined by a semi-major axis and a semi-minor axis.
                    The most common values for these two parameters are defined by
                    the World Geodetic Standard 1984 (WGS-84). The WGS-84 ellipsoid is
                    intended to correspond to mean sea level. A Geodetic height of zero
                    therefore roughly corresponds to sea level, with positive values increasing
                    away from the Earth’s center. The theoretical range of height values is
                    from the center of the Earth (about -6,371km) to positive infinity.

    Imp:
            It was necessary to divide __new__ into __new__ and __init__ to keep track
            of reference attribute. Using __new__ only caused reference to be a class
            attribute. So, if you change reference (as a class attribute) in any part
            of the code, reference for all instances changes resulting in a wrong calculation.
            To save reference as instance attribute instead of class attribute,
            __init__ is necessary. Same approach is used in LTP.
            Todo: There might be an elegant way to do this.
    """

    def __new__(
        cls,
        arg: Any = None,
        latitude: Union[float, int, np.ndarray] = None,
        longitude: Union[float, int, np.ndarray] = None,
        height: Union[float, int, np.ndarray] = None,
        *args,
        **kwargs,
    ):

        if isinstance(latitude, (float, np.ndarray)):
            return super().__new__(cls, latitude=latitude, longitude=longitude, height=height)
        elif not isinstance(arg, type(None)):
            if isinstance(arg, (LTP, ECEF, Geodetic, GRANDCS)):
                placeholder = np.nan * np.ones(len(arg[0]))
                return super().__new__(
                    cls, latitude=placeholder, longitude=placeholder, height=placeholder
                )
            else:
                raise TypeError(
                    type(arg),
                    "Argument type must be either \
						   ECEF, Geodetic, LTP, GRANDCS or Horizontal.",
                )
        else:
            # TODO: This part maynot be required.
            # return a placeholder with 1 entry. This is used if we just want to define LTP frame
            # without giving any coordinates. Can also use np.empty((1,1)) instead of np.array([nan]).
            # Do not use array with no entry because n>=1 is needed to instantiate 'Coordinates'.
            return super().__new__(
                cls,
                latitude=np.array([np.nan]),
                longitude=np.array([np.nan]),
                height=np.array([np.nan]),
            )

    def __init__(
        self,
        arg: Any = None,
        latitude: Union[float, int, np.ndarray] = None,
        longitude: Union[float, int, np.ndarray] = None,
        height: Union[float, int, np.ndarray] = None,
        reference: Any = "GEOID",
    ):  # options: 'GEOID', 'ELLIPSOID'
        """
        Create a new instance from another point instance or from
        latitude, longitude, height values
        """
        reference = reference.upper()
        self.reference = reference

        if not isinstance(arg, type(None)):
            arg = copy(arg)
            if isinstance(arg, Geodetic):
                arg = copy(arg)
                latitude, longitude = arg.latitude, arg.longitude
                # Use height wrt to ellipsoid or geoid (above sea level). Default is 'GEOID' (asl).
                if reference == "GEOID":
                    if arg.reference == "GEOID":
                        height = arg.height
                    elif arg.reference == "ELLIPSOID":
                        height = arg.height - geoid_undulation(
                            latitude=arg.latitude, longitude=arg.longitude
                        )
                elif reference == "ELLIPSOID":
                    if arg.reference == "GEOID":
                        height = arg.height + geoid_undulation(
                            latitude=arg.latitude, longitude=arg.longitude
                        )
                    elif arg.reference == "ELLIPSOID":
                        height = arg.height
                else:
                    raise TypeError(
                        "Provide reference as GEOID or ELLIPSOID istead of %s" % str(reference)
                    )

            elif isinstance(arg, Horizontal):
                # TO DO: write proper transformation.
                pass
            elif isinstance(arg, ECEF):
                # allows ECEF instance as an input. Convert it to Geodetic.
                latitude, longitude, height = turtle.ecef_to_geodetic(
                    arg.T
                )  # height here is wrt ellipsoid.
                if reference == "GEOID":
                    height = height - geoid_undulation(latitude=latitude, longitude=longitude)
                elif reference == "ELLIPSOID":
                    pass
                else:
                    raise TypeError(
                        "Provide reference as GEOID or ELLIPSOID istead of %s" % str(reference)
                    )
            elif isinstance(arg, (LTP, GRANDCS)):
                ecef = ECEF(arg)
                geodetic = Geodetic(ecef, reference=reference)
                latitude, longitude, height = (
                    geodetic.latitude,
                    geodetic.longitude,
                    geodetic.height,
                )
            else:
                raise TypeError(
                    type(arg),
                    type(latitude),
                    "Argument type must be either int, float, np.ndarray, \
							ECEF, Geodetic, GRANDCS or Horizontal.",
                )

        if isinstance(latitude, (Number, np.ndarray)):
            # use setter to replace placeholder coordinates values with the real values.
            self.latitude = latitude
            self.longitude = longitude
            self.height = height
        else:
            raise TypeError(
                type(latitude),
                "latitude, longitude, and height type must be either \
					   int, float, np.ndarray.",
            )

    def geodetic_to_horizontal(self):
        pass

    def geodetic_to_ecef(self):
        return ECEF(self)

    def geodetic_to_grandcs(self):
        return GRANDCS(self)

    def geodetic_to_ltp(self, ltp):
        ecef = ECEF(self)
        pos_v = np.vstack(
            (ecef.x - ltp.location.x, ecef.y - ltp.location.y, ecef.z - ltp.location.z)
        )
        ltp_cord = np.matmul(ltp.basis, pos_v)
        x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        return LTP(x=x, y=y, z=z, frame=ltp)


class ECEF(CartesianRepresentation):
    """
    Generic container for Earth-Centered Earth-Fixed (ECEF) coordinate system.
    Center of Earth is the origin of this frame.

    ECEF is a right-handed Cartesian coordinate system with the origin at the
    Earth’s center. This coordinate frame is fixed with respect to the Earth
    (i.e., rotates along with the Earth). Units are in meters. The three axis are
    defined as follows:
    x:	Passes through the equator at the Prime Meridian (latitude = 0, longitude = 0).
    y:	Passes through the equator 90 degrees east of the Prime Meridian
            (latitude = 0, longitude = 90 degrees).
    z:	Passes through the North Pole (latitude = 90 degrees, longitude = any value).

    Note:
            Reason to have __new__ and __init__ is to store obstime as instance attribute.
            obstime will be saved as class attribute if only __new__ is used. This means
            if obstime is modified for one case anywhere during calculation, obstime for
            all cases will be modified.
            Todo: find an elegant way to do this.
    """

    def __new__(
        cls,
        arg: Any = None,
        x: Union[float, int, np.ndarray] = None,
        y: Union[float, int, np.ndarray] = None,
        z: Union[float, int, np.ndarray] = None,
        *args,
        **kwargs,
    ):

        if isinstance(x, (Number, np.ndarray)):
            return super().__new__(cls, x=x, y=y, z=z)
        elif not isinstance(arg, type(None)):
            if isinstance(arg, (LTP, ECEF, Geodetic, GRANDCS)):
                placeholder = np.nan * np.ones(len(arg[0]))
                return super().__new__(cls, x=placeholder, y=placeholder, z=placeholder)
            else:
                raise TypeError(
                    type(arg),
                    "Argument type must be either \
						   ECEF, Geodetic, LTP, GRANDCS or Horizontal.",
                )
        else:
            # TODO: This part maynot be required.
            # return a placeholder with 1 entry. This is used if we just want to define LTP frame
            # without giving any coordinates. Can also use np.empty((1,1)) instead of np.array([nan]).
            # Do not use array with no entry because n>=1 is needed to instantiate 'Coordinates'.
            return super().__new__(
                cls, x=np.array([np.nan]), y=np.array([np.nan]), z=np.array([np.nan])
            )

    def __init__(
        self,
        arg: Any = None,
        x: Union[float, int, np.ndarray] = None,
        y: Union[float, int, np.ndarray] = None,
        z: Union[float, int, np.ndarray] = None,
        obstime: Union[str, datetime] = "2020-01-01",
    ):

        self.obstime = obstime

        if not isinstance(arg, type(None)):
            arg = copy(arg)
            if isinstance(arg, Horizontal):
                # TO DO: write a proper transformation from Horizontal to ECEF.
                pass
            elif isinstance(arg, ECEF):
                x, y, z = arg.x, arg.y, arg.z
            elif isinstance(arg, Geodetic):
                # allows Geodtic instances as input. Convert from Geodetic to ECEF.
                # change height wrt geoid to wrt ellipsoid.
                if arg.reference == "GEOID":
                    # height wrt ellipsoid
                    height = arg.height + geoid_undulation(
                        latitude=arg.latitude, longitude=arg.longitude
                    )
                elif (
                    arg.reference == "ELLIPSOID"
                ):  # leave it as it is because turtle uses height wrt ellipsoid.
                    height = arg.height
                ecef = turtle.ecef_from_geodetic(arg.latitude, arg.longitude, height)
                if ecef.size == 3:
                    x, y, z = ecef[0], ecef[1], ecef[2]
                elif ecef.size > 3:
                    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
            elif isinstance(arg, (LTP, GRANDCS)):
                basis = arg.basis
                origin = arg.location
                ecef = np.matmul(basis.T, arg) + origin
                x, y, z = ecef.x, ecef.y, ecef.z
            else:
                raise TypeError(
                    type(arg),
                    type(x),
                    "Type must be either int, float, np.ndarray, \
							ECEF, Geodetic, GRANDCS or Horizontal.",
                )

        if isinstance(x, (Number, np.ndarray)):
            # use setter to replace placeholder coordinates values with the real values.
            self.x = x
            self.y = y
            self.z = z
        else:
            raise TypeError(type(x), "x, y, and z type must be either int, float, np.ndarray.")

    def ecef_to_geodetic(self, reference="GEOID"):
        return Geodetic(self, reference=reference)

    def ecef_to_grandcs(self):
        return GRANDCS(self)

    def ecef_to_ltp(self, ltp):
        self = copy(self)
        pos_v = np.vstack(
            (self.x - ltp.location.x, self.y - ltp.location.y, self.z - ltp.location.z)
        )
        ltp_cord = np.matmul(ltp.basis, pos_v)
        x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        return LTP(x=x, y=y, z=z, frame=ltp)


grandcs_origin = Geodetic(
    latitude=grd_origin_lat,
    longitude=grd_origin_lon,
    height=grd_origin_height,
    reference="GEOID",
)

# RK: Merged into Horizontal. Delete this class.
class HorizontalVector(HorizontalRepresentation):
    """
    This class has been merged into Horizontal class
    by adding 'vector' attribute to reduce code duplication.
    """

    pass


# RK: Rework on this class
class Horizontal(HorizontalRepresentation):
    """
    Generic container for horizontal coordinates.

    azimuth  : angle (deg) starting from true North towards East.
    elevation: angle (deg) from horizontal plane towards zenith.
    """

    def __new__(
        cls,
        arg: Any = None,
        azimuth: Union[float, int, np.ndarray] = None,
        elevation: Union[float, int, np.ndarray] = None,
        norm: Union[float, int, np.ndarray] = 1.0,
        location: Any = grandcs_origin,
        vector: bool = False,
    ):
        """
        n: number of coordinate points. 3xn np.ndarray object will be instantiated
           which will then be replaced by input azimuth, elevation, and norm.
           'n' has to be predefined.
        location: origin of Horizontal coordinate system. Can be given in any known
                        coordinate system.
        """
        obj = LTP(location=location, orientation="ENU", magnetic=False)
        ecef_loc = obj.location  # location is already in ECEF cs.
        ecef_basis = obj.basis  # basis is already in ECEF cs.
        cls.location = ecef_loc  # used to convert back to ECEF, Geodetic etc.
        cls.basis = ecef_basis  # used to convert back to ECEF, Geodetic etc.
        cls.vector = vector

        if isinstance(arg, (Horizontal, HorizontalRepresentation)):
            return Horizontal(azimuth=arg.azimuth, elevation=arg.elevation, norm=arg.norm)

        if isinstance(azimuth, (Number, np.ndarray)):
            # check if input coordinates are of the right kind.
            # Additional check will be performed inside HorizontalRepresentation.
            pass
        elif not isinstance(arg, type(None)):
            if isinstance(arg, (ECEF, Geodetic, LTP, GRANDCS)):
                if isinstance(arg, ECEF):
                    ecef = arg  # No need to convert. ECEF is required.
                elif isinstance(arg, (Geodetic, LTP, GRANDCS)):
                    ecef = ECEF(arg)  # Convert from Geodetic input to ECEF.

                # Positional vector from the new location is used like for GRANDCS cs.
                if vector:
                    pos_v = np.vstack((ecef.x, ecef.y, ecef.z))
                else:
                    pos_v = np.vstack(
                        (
                            ecef.x - cls.location.x,
                            ecef.y - cls.location.y,
                            ecef.z - cls.location.z,
                        )
                    )
                # Projecting positional vectors to 'ENU' LTP's cs basis.
                # Below matrix multiplication is performed in turtle.ecef_to_horizontal()
                enu_cord = np.matmul(ecef_basis, pos_v)
                x, y, z = (
                    enu_cord[0],
                    enu_cord[1],
                    enu_cord[2],
                )  # x,y,z w.r.t to ENU basis.
                r = np.sqrt(x * x + y * y + z * z)
                azimuth = np.rad2deg(np.arctan2(x, y))
                elevation = np.rad2deg(np.arcsin(z / r))
                norm = r
            else:
                raise TypeError(
                    type(arg),
                    type(azimuth),
                    "Type must be either int, float, np.ndarray, \
							ECEF, Geodetic, GRANDCS or Horizontal.",
                )
        else:
            raise TypeError(
                type(arg),
                type(azimuth),
                "Type must be either int, float, np.ndarray, \
						ECEF, Geodetic, GRANDCS or Horizontal.",
            )

        return super().__new__(cls, azimuth, elevation, norm)

    def horizontal_to_ecef(self):
        rel = np.deg2rad(self.elevation)
        raz = np.deg2rad(self.azimuth)
        ce = np.cos(rel)

        pos_v = np.vstack(
            (
                self.norm * ce * np.sin(raz),
                self.norm * ce * np.cos(raz),
                self.norm * np.sin(rel),
            )
        )
        # Projecting horizontal direction vectors to ECEF's cs basis.
        # Converts completely from Horizontal to ECEF cs.
        # Below matrix multiplication is performed in turtle.ecef_to_horizontal()
        if self.vector:
            ecef_cord = np.matmul(self.basis.T, pos_v)  # basis is in ECEF frame.
        else:
            ecef_cord = np.matmul(self.basis.T, pos_v) + self.location  # basis is in ECEF frame.
        x, y, z = ecef_cord[0], ecef_cord[1], ecef_cord[2]  # x,y,z w.r.t to ECEF basis.

        return ECEF(x=x, y=y, z=z)

    def horizontal_to_geodetic(self):
        ecef = self.horizontal_to_ecef()
        return Geodetic(ecef)

    def horizontal_to_grandcs(self):
        ecef = self.horizontal_to_ecef()
        return GRANDCS(ecef)


class LTP(CartesianRepresentation):
    """
    Calculates basis and orgin at a given latitude and longitude.
    Basis and origin is calculated in ECEF frame.
    'location' and 'orientation' are required.
    """

    def __new__(
        cls,
        arg: Any = None,  # input coordinate instance to convert to LTP
        x: Union[float, int, np.ndarray] = None,  # x-coordinate at LTP
        y: Union[float, int, np.ndarray] = None,  # y-coordinate at LTP
        z: Union[float, int, np.ndarray] = None,  # z-coordinate at LTP
        *args,
        **kwargs,
    ):

        if isinstance(x, (Number, np.ndarray)):
            return super().__new__(cls, x=x, y=y, z=z)
        elif not isinstance(arg, type(None)):
            if isinstance(arg, (LTP, ECEF, Geodetic, GRANDCS)):
                if isinstance(arg, ECEF):
                    ecef = arg  # No need to convert. ECEF is required.
                elif isinstance(arg, (LTP, Geodetic, GRANDCS)):
                    ecef = ECEF(arg)  # Convert from Geodetic input to ECEF.
                placeholder = np.nan * np.ones(len(ecef.x))
                return super().__new__(cls, x=placeholder, y=placeholder, z=placeholder)
            else:
                raise TypeError(
                    type(arg),
                    "Argument type must be either \
						   ECEF, Geodetic, LTP, GRANDCS or Horizontal.",
                )
        else:
            # return a placeholder with 1 entry. This is used if we just want to define LTP frame
            # without giving any coordinates. Can also use np.empty((1,1)) instead of np.array([nan]).
            # Do not use array with no entry because n>=1 is needed to instantiate 'Coordinates'.
            return super().__new__(
                cls, x=np.array([np.nan]), y=np.array([np.nan]), z=np.array([np.nan])
            )

    def __init__(
        self,
        arg: Any = None,  # input coordinate instance to convert to LTP
        x: Union[float, int, np.ndarray] = None,  # x-coordinate at LTP.
        y: Union[float, int, np.ndarray] = None,  # y-coordinate at LTP
        z: Union[float, int, np.ndarray] = None,  # z-coordinate at LTP
        latitude: Union[float, int, np.ndarray] = None,  # latitude of LTP's location/origin
        longitude: Union[float, int, np.ndarray] = None,  # longitude of LTP's location/origin
        height: Union[float, int, np.ndarray] = None,  # height of LTP's location/origin
        reference: str = None,  # reference for Geodetic location
        location: Any = None,  # location of LTP in Geodetic, GRANDCS, or ECEF
        orientation: str = None,  # orientation of LTP. 'NWU', 'ENU' etc
        magnetic: bool = False,  # shift orientation by magnetic declination?
        magmodel: str = "IGRF13",  # if shift, which magnetic model to use?
        declination: Union[float, np.ndarray] = None,  # or simply provide the magnetic declination
        obstime: Union[str, datetime] = "2020-01-01",  # calculate declination of what date?
        frame: Any = None,
        rotation=None,
    ):

        # Make sure the location is in the correct format. i.e ECEF, Geodetic, GeodeticRepresentation,
        # or GRANDCS cs. OR latitude=deg, longitude=deg, height=meter.
        if frame is not None:
            frame = copy(frame)
            geodetic_loc = (
                frame.location if isinstance(frame.location, Geodetic) else Geodetic(frame.location)
            )
            orientation = frame.orientation
            magnetic = frame.magnetic
            declination = frame.declination
            magmodel = frame.magmodel
            obstime = frame.obstime
        elif latitude != None and longitude != None and height != None:
            geodetic_loc = Geodetic(
                latitude=latitude,
                longitude=longitude,
                height=height,
                reference=reference,
            )
        elif isinstance(location, Geodetic):
            geodetic_loc = location  # This is to preserve reference.
        elif isinstance(location, (LTP, ECEF, GeodeticRepresentation, GRANDCS)):
            geodetic_loc = Geodetic(location)  # default GEOID reference is used.
        else:
            raise TypeError(
                "Provide location of LTP in ECEF, Geodetic, or GRANDCS coordinate system instead of type %s.\n \
							Location can also be given as latitude=deg, longitude=deg, height=meter."
                % type(location)
            )

        # Make sure orientation is given as string.
        if isinstance(orientation, str):
            pass
        else:
            raise TypeError(
                "Provide orientaion. \
				Orientation must be string instead of %s. Example: ENU, NWU etc."
                % type(orientation)
            )

        latitude = geodetic_loc.latitude
        longitude = geodetic_loc.longitude
        height = geodetic_loc.height

        # Calculate magnetic field declination if magnetic=True. Used to define GRANDCS coordinate system.
        if magnetic and declination is None:
            from .geomagnet import Geomagnet

            # Calculate a magnetic field declination at a given location at a give time.
            geoB = Geomagnet(magmodel, location=geodetic_loc, obstime=obstime)
            declination = geoB.declination

        azimuth0 = 0.0 if declination is None else declination
        magnetic = magnetic if declination is None else True

        def vector(name):
            tag = name[0].upper()
            if tag == "E":
                return turtle.ecef_from_horizontal(latitude, longitude, 90 + azimuth0, 0)
            elif tag == "W":
                return turtle.ecef_from_horizontal(latitude, longitude, 270 + azimuth0, 0)
            elif tag == "N":
                return turtle.ecef_from_horizontal(latitude, longitude, azimuth0, 0)
            elif tag == "S":
                return turtle.ecef_from_horizontal(latitude, longitude, 180 + azimuth0, 0)
            elif tag == "U":
                return turtle.ecef_from_horizontal(latitude, longitude, 0, 90)
            elif tag == "D":
                return turtle.ecef_from_horizontal(latitude, longitude, 0, -90)

            else:
                raise ValueError(f"Invalid frame orientation `{name}`")

        # unit vectors (basis) in ECEF frame of reference.
        # These are the basis of the GRANDCS coordinate system if orientation='NWU' and magnetic=True.
        ux = vector(orientation[0])
        uy = vector(orientation[1])
        uz = vector(orientation[2])

        # These objects share the same memory with arg and is overwritten if kept inside __new__.
        # Problem is solved if __new__ is redefined and __init__ is added with below attributes
        # in __init__ rather than in __new__.
        self.location = ECEF(geodetic_loc)
        self.basis = np.vstack((ux, uy, uz))  # unit vectors (basis) in ECEF frame.
        self.orientation = orientation
        self.magnetic = magnetic
        self.declination = azimuth0
        self.magmodel = magmodel
        self.obstime = obstime
        self.rotation = rotation

        # Scripts below is used only if coordinates (x,y,z) in LTP's frame is required.
        if not isinstance(arg, type(None)):
            arg = copy(arg)
            if isinstance(arg, (LTP, ECEF, Geodetic, GRANDCS)):
                if isinstance(arg, ECEF):
                    ecef = arg  # No need to convert. ECEF is required.
                elif isinstance(arg, (LTP, Geodetic, GRANDCS)):
                    ecef = ECEF(arg)  # Convert from Geodetic input to ECEF.
                # Positional vectors wrt GRANDCS's cs origin. Still in ECEF cs.
                pos_v = np.vstack(
                    (
                        ecef.x - self.location.x,
                        ecef.y - self.location.y,
                        ecef.z - self.location.z,
                    )
                )
                # Projecting positional vectors to LTP's basis.
                # Converts completely from ECEF cs to the LTP's frame.
                ltp_cord = np.matmul(self.basis, pos_v)
                x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        if isinstance(x, (Number, np.ndarray)):
            # use setter to replace placeholder coordinates values with the real values.
            self.x = x
            self.y = y
            self.z = z

    def ltp_to_ltp(self, ltp):
        # convert self to ECEF frame. Then convert ecef to new ltp's frame.
        ecef = ECEF(self)
        pos_v = np.vstack(
            (ecef.x - ltp.location.x, ecef.y - ltp.location.y, ecef.z - ltp.location.z)
        )
        ltp_cord = np.matmul(ltp.basis, pos_v)
        x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        return LTP(x=x, y=y, z=z, frame=ltp)

    def ltp_to_grandcs(self):
        # just instantiating a GRANDCS CS to get it's basis and location. x, y, z values does not matter.
        self = copy(self)
        gcs = GRANDCS(x=0, y=0, z=0)
        self.ltp_to_ltp(gcs)

    def ltp_to_ecef(self):
        # Basis forms a rotational matrix. Transpose is the inverse of rotational matrix (real).
        # Use inverse (transpose) of rotational matrix to convert from GRANDCS to ECEF.
        return ECEF(self)

    def ltp_to_geodetic(self, reference="GEOID"):
        # Convert from GRANDCS to ECEF, then from ECEF to Geodetic.
        ecef = ECEF(self)
        return Geodetic(ecef, reference=reference)


class GRANDCS(LTP):
    """
    Class for the GRANDCS coordinate system (cs). This class instantiate coordinates
    in GRANDCS's coordinate frame. Input can be either x, y, z coordinates value
    in GRANDCS's cs or coordinates in ECEF or Geodetic system.

    If the input coordinates are in ECEF or Geodetic (or any coordinate system
    other than GRANDCS's cs) following procedure is performed. Convert everything
    to ECEF frame to do all initial calculations and then finally project the
    positional vector from GRANDCS's cs origin to it's basis vectors. Basis for NWU GRANDCS's
    cs is calculated based on GRANDCS's origin. Basis forms a rotational matrix.
    Transpose of the rotational matrix is it's inverse. Basis in this case are unit
    vectors in ECEF frame that describes the NWU direction from GRANDCS's origin
    (also in ECEF frame). To convert other cs to GRANDCS's cs, first transform the
    coordinates in ECEF frame, then shift the origin of coordinates from
    ECEF's origin to GRANDCS's origin. Now you get a positional vector from GRANDCS's origin
    in ECEF frame. Take a dot product of this positional vector with the NWU basis unit
    vectors to get the coordinates in GRANDCS coordinate system.

    Use inverse (transpose) of rotational matrix to convert from GRANDCS cs to ECEF. Then
    convert from ECEF to Geodetic.
    """

    def __init__(
        self,
        arg: Any = None,
        x: Union[float, int, np.ndarray] = None,
        y: Union[float, int, np.ndarray] = None,
        z: Union[float, int, np.ndarray] = None,
        latitude: Union[float, int, np.ndarray] = None,  # latitude of LTP's location/origin
        longitude: Union[float, int, np.ndarray] = None,  # longitude of LTP's location/origin
        height: Union[float, int, np.ndarray] = None,  # height of LTP's location/origin
        location: Any = grandcs_origin,
        obstime: Union[str, datetime] = "2020-01-01",
        rotation=None,
    ):

        # Added for tests.
        if arg is not None:
            if isinstance(arg, (ECEF, Horizontal, Geodetic, LTP, GRANDCS)):
                pass
            else:
                raise TypeError(
                    type(arg),
                    "Argument type must be \
							ECEF, Geodetic, LTP, GRANDCS or Horizontal.",
                )
        if x is not None:
            if isinstance(x, (Number, np.ndarray)):
                pass
            else:
                raise TypeError(type(x), "x type must be int, float or np.ndarray.")

        super().__init__(
            arg=arg,  # input coordinate instance to convert to LTP
            x=x,  # x-coordinate at LTP.
            y=y,  # y-coordinate at LTP
            z=z,  # z-coordinate at LTP
            latitude=latitude,  # latitude of LTP's location/origin
            longitude=longitude,  # longitude of LTP's location/origin
            height=height,  # height of LTP's location/origin
            location=location,  # location of LTP in Geodetic, GRANDCS, or ECEF
            orientation="NWU",  # orientation of LTP. 'NWU', 'ENU' etc
            magnetic=True,  # shift orientation by magnetic declination?
            magmodel="IGRF13",  # if shift, which magnetic model to use?
            declination=None,  # or simply provide the magnetic declination
            obstime=obstime,  # calculate declination of what date?
            rotation=rotation,
        )

    def grandcs_to_ecef(self):
        # Basis forms a rotational matrix. Transpose is the inverse of rotational matrix (real).
        # Use inverse (transpose) of rotational matrix to convert from GRANDCSCS to ECEF.
        return self.ltp_to_ecef()

    def grandcs_to_geodetic(self, reference="GEOID"):
        # Convert from GRANDCSCS to ECEF, then from ECEF to Geodetic.
        return self.ltp_to_geodetic(reference=reference)

    def grandcs_to_ltp(self, ltp):
        # Convert from GRANDCSCS to ECEF, then from ECEF to Geodetic.
        return self.ltp_to_ltp(ltp)


# RK TODO: Rework on this class.
class Rotation(_Rotation):
    pass
