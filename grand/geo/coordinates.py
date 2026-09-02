"""Terrestrial coordinate systems and the conversions between them.

The frames are :class:`Geodetic`, :class:`ECEF`, :class:`LTP` and
:class:`GRANDCS`; every conversion between a local frame and geodetic passes
through ECEF.  See :doc:`/coordinates` for the conventions, including the two
that most often catch people out: every angle is in degrees, and GRANDCS
:math:`x` runs north where an ENU :math:`x` runs east.

units: 

 *	energy: GeV
 * time  : nanosecond
 *	length: m
 *	voltage: microVolt
 * 	Efield: microVolt/m
 *	Bfield: nanoTesla
 *	angles: degree
 *	grammage: g/cm2
 *	Density: g/cm3 
 *	frequency: Hz
 *	resistance: Ohm.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Any
from typing_extensions import Final
from datetime import datetime
import copy as _copy
import enum
import os
from numbers import Number
from logging import getLogger
import warnings
import numpy as np
try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None
from . import turtle
from grand import grand_get_path_root_pkg

logger = getLogger(__name__)
# add protection against casting complex to real. Need to try due to numpy versions incompatibilities
try:
    warnings.filterwarnings(action="error", category=np.ComplexWarning)
# After numpy 1.25. In principle the np.ComplexWarning should still be accessible, but in 2.2.5 it isn't
except:
    warnings.filterwarnings(action="error", category=np.exceptions.ComplexWarning)

DATADIR: Final = grand_get_path_root_pkg() + "/data"  # for geoid_undulation egm96.png file.

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
    r"""Returns a copy of `obj`, with its instance attributes copied too.

    ``numpy.ndarray`` subclasses do not carry their instance dictionary
    through :func:`copy.copy`, so the frame attributes a coordinate object
    holds -- its origin, its basis, its reference level -- would be lost.
    This copies the array and then the attributes.

    Parameters
    ----------
    obj : object
        The object to copy, typically a :class:`Coordinates` subclass.
    deep : bool, optional
        Copy recursively with :func:`copy.deepcopy` instead of shallowly.
    attributes : list, optional
        Unused; retained because callers pass it.

    Returns
    -------
    object
        A copy of `obj`, of the same type.
    """
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
    r"""Reference surface for a height in the geodetic system.

    A geodetic height is meaningless without saying what it is measured
    from.  ``ELLIPSOID`` measures from the WGS-84 ellipsoid, the smooth
    mathematical figure; ``GEOID`` measures from mean sea level, which
    departs from the ellipsoid by up to about 100 m.  Confusing the two is
    a common source of vertical errors at the array.

    .. versionadded:: 0.1.0

    Notes
    -----
    The same enumeration is defined in :mod:`grand.geo.topography`.  It is
    repeated here because importing it would make the two modules circular.
    """

    ELLIPSOID = enum.auto()
    GEOID = enum.auto()


def geoid_undulation(latitude=None, longitude=None):
    r"""Returns the height of the geoid above the ellipsoid, in metres.

    The geoid undulation is what converts between the two references of
    :class:`Reference`: a height above mean sea level plus the undulation
    at that location is the height above the ellipsoid.  Values are read
    from the EGM96 model shipped in ``data/egm96.png``.

    Parameters
    ----------
    latitude : float or ndarray
        Geodetic latitude, in degrees.
    longitude : float or ndarray
        Geodetic longitude, in degrees.

    Returns
    -------
    float or ndarray
        Undulation in metres, positive where the geoid lies above the
        ellipsoid.

    Examples
    --------
    .. jupyter-execute::

        from grand.geo.coordinates import geoid_undulation

        # The GRANDProto300 site at Dunhuang.
        print("%.2f m" % geoid_undulation(latitude=40.98, longitude=93.95))

    Notes
    -----
    Also defined in :mod:`grand.geo.topography`; repeated here to avoid a
    circular import.
    """
    path = os.path.join(DATADIR, "egm96.png")
    geoid = turtle.Map(path)
    logger.debug(f"geoid_undulation for {latitude} {longitude}")
    return geoid.elevation(longitude, latitude)


# Define functions to transform from one coordinate representation to
# another coordinate representation. Cartesian, Spherical, and Horizontal
# coordinate representation are defined.
def _cartesian_to_spherical(
    x: Union[float, int, np.ndarray],
    y: Union[float, int, np.ndarray],
    z: Union[float, int, np.ndarray],
) -> Union[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    r"""Returns the spherical coordinates of a Cartesian vector.

    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian components, in the same length unit.

    Returns
    -------
    theta : float or ndarray
        Polar angle from the :math:`+z` axis, in **degrees**, in
        :math:`[0, 180]`.
    phi : float or ndarray
        Azimuth from the :math:`+x` axis towards :math:`+y`, in
        **degrees**, in :math:`(-180, 180]`.
    r : float or ndarray
        Radius, in the unit of the inputs.

    Examples
    --------
    .. jupyter-execute::

        from grand.geo.coordinates import _cartesian_to_spherical

        print(_cartesian_to_spherical(0.0, 0.0, 1.0))   # straight up
        print(_cartesian_to_spherical(1.0, 0.0, 0.0))   # on the +x axis

    Notes
    -----
    Angles are in degrees, not radians, throughout this module.
    """
    rho2 = x ** 2 + y ** 2
    rho = np.sqrt(rho2)
    theta = np.rad2deg(np.arctan2(rho, z))
    phi = np.rad2deg(np.arctan2(y, x))
    r = np.sqrt(rho2 + z ** 2)

    return theta, phi, r


# Horizontal has an axis fixed to geographic North, so it can not be
# converted like cartesian and spherical. If conversion is done, then
# the same origin for both and ENU basis for cartesian is assumed.
def _cartesian_to_horizontal(
    x: Union[float, int, np.ndarray],
    y: Union[float, int, np.ndarray],
    z: Union[float, int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""Returns the horizontal coordinates of a Cartesian vector.

    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian components in an **ENU** basis -- :math:`x` east,
        :math:`y` north, :math:`z` up -- sharing the horizontal frame's
        origin.

    Returns
    -------
    azimuth : float or ndarray
        Degrees, measured from geographic north towards east.
    elevation : float or ndarray
        Degrees above the horizon.
    norm : float or ndarray
        Length of the vector.

    Notes
    -----
    The horizontal frame has an axis fixed to geographic north, so unlike
    the Cartesian and spherical representations it is not a pure change of
    variables: the ENU basis and a shared origin are assumed.  Passing
    components expressed in some other Cartesian basis silently gives the
    wrong azimuth.
    """
    theta, phi, r = _cartesian_to_spherical(x, y, z)
    return _spherical_to_horizontal(theta, phi, r)


def _spherical_to_cartesian(
    theta: Union[float, int, np.ndarray],
    phi: Union[float, int, np.ndarray],
    r: Union[float, int, np.ndarray],
) -> Union[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    r"""Returns the Cartesian components of a spherical vector.

    Parameters
    ----------
    theta : float or ndarray
        Polar angle from :math:`+z`, in **degrees**.
    phi : float or ndarray
        Azimuth from :math:`+x`, in **degrees**.
    r : float or ndarray
        Radius.

    Returns
    -------
    x, y, z : float or ndarray
        Cartesian components, in the unit of `r`.

    Examples
    --------
    Round-tripping recovers the input, which is the property every test of
    this pair relies on:

    .. jupyter-execute::

        import numpy as np
        from grand.geo.coordinates import (_cartesian_to_spherical,
                                           _spherical_to_cartesian)

        v = (3.0, -4.0, 12.0)
        back = _spherical_to_cartesian(*_cartesian_to_spherical(*v))
        print(np.round(back, 12))
    """
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
    r"""Returns the horizontal coordinates of a spherical vector.

    Parameters
    ----------
    theta : float or ndarray
        Polar angle from :math:`+z`, in **degrees**.
    phi : float or ndarray
        Azimuth from :math:`+x`, in **degrees**.
    r : float or ndarray
        Radius.

    Returns
    -------
    azimuth : float or ndarray
        :math:`90 - \phi`, in degrees: measured from north towards east
        rather than from :math:`+x` towards :math:`+y`.
    elevation : float or ndarray
        :math:`90 - \theta`, in degrees: measured up from the horizon
        rather than down from the zenith.
    norm : float or ndarray
        Unchanged radius.

    Examples
    --------
    .. jupyter-execute::

        from grand.geo.coordinates import _spherical_to_horizontal

        # Straight up: zero polar angle becomes 90 degrees of elevation.
        print(_spherical_to_horizontal(0.0, 0.0, 1.0))
    """
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
    r"""Returns the Cartesian components of a horizontal vector.

    Parameters
    ----------
    azimuth : float or ndarray
        Degrees from geographic north towards east.
    elevation : float or ndarray
        Degrees above the horizon.
    norm : float or ndarray
        Length of the vector.

    Returns
    -------
    x, y, z : float or ndarray
        Components in an ENU basis, in the unit of `norm`.
    """
    theta, phi, r = _horizontal_to_spherical(azimuth, elevation, norm)
    return _spherical_to_cartesian(theta, phi, r)


def _horizontal_to_spherical(
    azimuth: Union[float, int, np.ndarray],
    elevation: Union[float, int, np.ndarray],
    norm: Union[float, int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""Returns the spherical coordinates of a horizontal vector.

    Parameters
    ----------
    azimuth : float or ndarray
        Degrees from geographic north towards east.
    elevation : float or ndarray
        Degrees above the horizon.
    norm : float or ndarray
        Length of the vector.

    Returns
    -------
    theta : float or ndarray
        :math:`90 - \mathrm{elevation}`, in degrees.
    phi : float or ndarray
        :math:`90 - \mathrm{azimuth}`, in degrees.
    r : float or ndarray
        Unchanged length.
    """
    # return 0.5 * np.pi - elevation, 0.5 * np.pi - azimuth, norm
    return 90.0 - elevation, 90.0 - azimuth, norm


# -----------Base Representation------------
class Coordinates(np.ndarray):
    """
    Generic container for a coordinates object.

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
    """Generic container for cartesian coordinates."""

    def __new__(
        cls,
        arg: Any = None,
        x: Union[float, np.ndarray] = None,
        y: Union[float, np.ndarray] = None,
        z: Union[float, np.ndarray] = None,
    ):
        """
        Create a Cartesian coordinates instance.

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

    def info(self):
        r"""Prints a short description of the object, for interactive use.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        ret = f"CartesianRepresentation: shape {self.shape}, min:{np.min(self)} max:{np.max(self)}"
        return ret

    @property
    def x(self):
        r"""Cartesian :math:`x` component, in metres."""
        return self[0]

    @x.setter
    def x(self, v):
        r"""Sets the :math:`x` component."""
        self[0] = v

    @property
    def y(self):
        r"""Cartesian :math:`y` component, in metres."""
        return self[1]

    @y.setter
    def y(self, v):
        r"""Sets the :math:`y` component."""
        self[1] = v

    @property
    def z(self):
        r"""Cartesian :math:`z` component, in metres."""
        return self[2]

    @z.setter
    def z(self, v):
        r"""Sets the :math:`z` component."""
        self[2] = v

    def cartesian_to_spherical(self):
        r"""Returns this vector as a :class:`SphericalRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        theta, phi, r = _cartesian_to_spherical(self[0], self[1], self[2])
        return SphericalRepresentation(theta=theta, phi=phi, r=r)

    def cartesian_to_horizontal(self):
        r"""Returns this vector as a :class:`HorizontalRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        azi, ele, norm = _cartesian_to_horizontal(self[0], self[1], self[2])
        return HorizontalRepresentation(azimuth=azi, elevation=ele, norm=norm)

    def norm(self):
        r"""Returns the Euclidean length of the vector, in metres.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        return np.linalg.norm(self)


class SphericalRepresentation(Coordinates):
    """Generic container for spherical coordinates."""

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
        r"""Polar angle from the :math:`+z` axis, in degrees."""
        logger.debug(f"{type(self)} {type(self[0])}")
        # TODO: self[0] and self have same type !!!!!
        # use float(self[0]) instead self[0] ?
        # return float(self[0])
        return self[0]

    @theta.setter
    def theta(self, v):
        r"""Sets the polar angle, in degrees."""
        self[0] = v

    @property
    def phi(self):
        r"""Azimuth from the :math:`+x` axis, in degrees."""
        # return float(self[1])
        return self[1]

    @phi.setter
    def phi(self, v):
        r"""Sets the azimuth, in degrees."""
        self[1] = v

    @property
    def r(self):
        r"""Radius, in metres."""
        return self[2]

    @r.setter
    def r(self, v):
        r"""Sets the radius, in metres."""
        self[2] = v

    def spherical_to_cartesian(self):
        r"""Returns this vector as a :class:`CartesianRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        x, y, z = _spherical_to_cartesian(self[0], self[1], self[2])
        return CartesianRepresentation(x=x, y=y, z=z)

    def spherical_to_horizontal(self):
        r"""Returns this vector as a :class:`HorizontalRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        azi, ele, norm = _spherical_to_horizontal(self[0], self[1], self[2])
        return HorizontalRepresentation(azimuth=azi, elevation=ele, norm=norm)


class HorizontalRepresentation(Coordinates):
    """Generic container for horizontal coordinates."""

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
        r"""Azimuth from geographic north towards east, in degrees."""
        return self[0]

    @azimuth.setter
    def azimuth(self, v):
        r"""Sets the azimuth, in degrees."""
        self[0] = v

    @property
    def elevation(self):
        r"""Elevation above the horizon, in degrees."""
        return self[1]

    @elevation.setter
    def elevation(self, v):
        r"""Sets the elevation, in degrees."""
        self[1] = v

    @property
    def norm(self):
        r"""Length of the vector, in metres."""
        return self[2]

    @norm.setter
    def norm(self, v):
        r"""Sets the length, in metres."""
        self[2] = v

    def horizontal_to_cartesian(self):
        r"""Returns this vector as a :class:`CartesianRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        x, y, z = _horizontal_to_cartesian(self[0], self[1], self[2])
        return CartesianRepresentation(x=x, y=y, z=z)

    def horizontal_to_spherical(self):
        r"""Returns this vector as a :class:`SphericalRepresentation`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        th, phi, r = _horizontal_to_spherical(self[0], self[1], self[2])
        return SphericalRepresentation(theta=th, phi=phi, r=r)


class GeodeticRepresentation(Coordinates):
    """
    Generic container for Geodetic coordinate system. Center of this frame.

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
        """Create a new instance from latitude, longitude, and height."""
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
        r"""Geodetic latitude, in degrees."""
        return self[0]

    @latitude.setter
    def latitude(self, v):
        r"""Sets the latitude, in degrees."""
        self[0] = v

    @property
    def longitude(self):
        r"""Geodetic longitude, in degrees."""
        return self[1]

    @longitude.setter
    def longitude(self, v):
        r"""Sets the longitude, in degrees."""
        self[1] = v

    @property
    def height(self):
        r"""Height above the reference surface, in metres."""
        return self[2]

    @height.setter
    def height(self, v):
        r"""Sets the height, in metres."""
        self[2] = v


# ------------------Frame---------------------
class Geodetic(GeodeticRepresentation):
    """
    Generic container for Geodetic coordinate system. Center of this frame.

    is the center of Earth.

    Latitude:	Angle north and south of the equator. +ve in the northern hemisphere,
                            -ve in the southern hemisphere. Range: -90 deg (South Pole)
                            to +90 deg (North Pole). In equator, latitude = 0.
    Longitude:	Angle east and west of the Prime Meridian. The Prime Meridian
                            is a north-south line that passes through Greenwich, UK.
                            +ve to the east of the Prime Meridian, -ve to the west.
                            Range: 0 deg to 360 deg positive or negative. 
                            Note that coordinate transformation is possible for +ve 0 to 360 deg.
                            So negative values are changed to positive by adding 360.
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
        r"""Returns a geodetic position, from components or another frame.

        Called either with `latitude`, `longitude` and `height`, or with a
        single positional `arg` holding a position in another frame, which is
        then converted.

        Parameters
        ----------
        arg : ECEF, LTP, GRANDCS or Geodetic, optional
            A position to convert.  Conversion passes through
            :class:`ECEF`.
        latitude : float or ndarray, optional
            Degrees north of the equator, in :math:`[-90, 90]`.
        longitude : float or ndarray, optional
            Degrees east of the prime meridian.
        height : float or ndarray, optional
            Metres above the reference surface -- see :class:`Reference`, and
            note that a height is meaningless without knowing which surface.

        Returns
        -------
        Geodetic
            The position.

        Examples
        --------
        .. jupyter-execute::

            import numpy as np
            from grand import Geodetic, ECEF

            site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
            print(np.round(np.asarray(Geodetic(ECEF(site))).ravel(), 6))
        """
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
        Create a new instance from another point instance or from.

        latitude, longitude, height values.
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
            # RK: +ve 0 to 360 were only accepted for coordinate transformation. 
            #     Now both +ve and -ve values are accepted for longitudes.
            if isinstance(latitude, Number):
                longitude = 360+longitude if longitude<0 else longitude
            else:
                longitude[longitude < 0] += 360

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
        r"""Returns this position in horizontal coordinates.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        pass

    def geodetic_to_ecef(self):
        r"""Returns this position in the :class:`ECEF` frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        return ECEF(self)

    def geodetic_to_grandcs(self):
        r"""Returns this position in the :class:`GRANDCS` array frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        return GRANDCS(self)

    def geodetic_to_ltp(self, ltp):
        r"""Returns this position in a local tangent plane, :class:`LTP`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        ecef = ECEF(self)
        pos_v = np.vstack(
            (ecef.x - ltp.location.x, ecef.y - ltp.location.y, ecef.z - ltp.location.z)
        )
        ltp_cord = np.matmul(ltp.basis, pos_v)
        x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        return LTP(x=x, y=y, z=z, frame=ltp)


class ECEF(CartesianRepresentation):
    r"""Earth-Centred, Earth-Fixed Cartesian coordinates, in metres.

    A right-handed frame whose origin is the centre of the Earth and which
    rotates with it.  Every conversion between a local frame and geodetic
    coordinates passes through ECEF; it is the common pivot, so that the
    ellipsoid constants live in one place.

    The axes are:

    - :math:`x`, through the equator at the prime meridian
      (latitude 0, longitude 0);
    - :math:`y`, through the equator 90 degrees east
      (latitude 0, longitude 90);
    - :math:`z`, through the north pole (latitude 90).

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from grand import Geodetic, ECEF

        site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
        print(np.round(np.asarray(ECEF(site)).ravel(), 1))

    See Also
    --------
    Geodetic : latitude, longitude and height.
    LTP : a local tangent plane.
    GRANDCS : the array frame.
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
        r"""Returns an Earth-centred, Earth-fixed vector.

        Parameters
        ----------
        arg : Geodetic, LTP, GRANDCS or ECEF, optional
            A position to convert.  ECEF is the pivot every other frame
            converts through.
        x, y, z : float or ndarray, optional
            Components in metres from the geocentre.

        Returns
        -------
        ECEF
            The vector.
        """
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
        r"""Initialises the frame and records the observation time.

        Parameters
        ----------
        arg : Geodetic, LTP, GRANDCS or ECEF, optional
            A position to convert.
        x, y, z : float or ndarray, optional
            Components in metres.
        obstime : str or datetime, optional
            Date the coordinates refer to.  It matters because the
            geomagnetic field, and any magnetic-north orientation derived
            from it, changes with time.
        """
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
        r"""Returns this position as latitude, longitude and height.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        return Geodetic(self, reference=reference)

    def ecef_to_grandcs(self):
        r"""Returns this position in the :class:`GRANDCS` array frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        return GRANDCS(self)

    def ecef_to_ltp(self, ltp):
        r"""Returns this position in a local tangent plane, :class:`LTP`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
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
    """Deprecated alias, merged into :class:`Horizontal`.

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
        n: number of coordinate points. 3xn np.ndarray object will be instantiated.

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
        r"""Returns this direction in the :class:`ECEF` frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
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
        r"""Returns this direction as a geodetic position.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        ecef = self.horizontal_to_ecef()
        return Geodetic(ecef)

    def horizontal_to_grandcs(self):
        r"""Returns this direction in the :class:`GRANDCS` array frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
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
        r"""Returns a local-tangent-plane vector, from components or another frame.

        Parameters
        ----------
        arg : ECEF, LTP, GRANDCS or Geodetic, optional
            A position to convert into this frame.
        x, y, z : float or ndarray, optional
            Components in metres, along the axes named by the frame's
            ``orientation``.

        Returns
        -------
        LTP
            The vector.

        Notes
        -----
        The axes depend on ``orientation``, so the same three numbers mean
        different places in different frames: ``'ENU'`` puts :math:`x` east,
        while :class:`GRANDCS` puts it north.  See :doc:`/coordinates`.
        """
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
        r"""Initialises the local frame: its origin, axes and epoch.

        Parameters
        ----------
        arg : Geodetic, ECEF, LTP or GRANDCS, optional
            A position to convert into this frame.
        x, y, z : float or ndarray, optional
            Components in metres along the frame's own axes.
        location : Geodetic, ECEF, LTP or GRANDCS, optional
            Origin of the frame.  A local frame without an origin cannot be
            converted to any other.
        orientation : str, optional
            Three characters, one per axis, from ``E``/``W``, ``N``/``S`` and
            ``U``/``D``.  ``'ENU'`` is east-north-up.
        magnetic : bool, optional
            Measure the horizontal axes from magnetic north rather than
            geographic north.  The declination is a few degrees at Dunhuang,
            which is hundreds of metres across a 10 km array, so this is a
            choice to make deliberately.
        obstime : str or datetime, optional
            Date used to evaluate the declination when `magnetic` is true.

        Notes
        -----
        See :doc:`/coordinates` for why the same components mean different
        places under different orientations.
        """
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
        elif latitude is not None and longitude is not None and height is not None:
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
            r"""Returns the ECEF direction of one named local axis.

            Parameters
            ----------
            name : str
                One axis of an orientation string: ``E``, ``W``, ``N``,
                ``S``, ``U`` or ``D``.  Only the first character is read,
                and case is ignored.

            Returns
            -------
            ndarray
                Unit vector in ECEF, shape ``(3,)``.

            Notes
            -----
            Horizontal axes are offset by ``azimuth0``, which is the
            magnetic declination when the frame was asked for magnetic
            north and zero otherwise.  This is where a ``magnetic=True``
            frame stops agreeing with a geographic one.
            """
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
        r"""Returns this vector in another local tangent plane.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # convert self to ECEF frame. Then convert ecef to new ltp's frame.
        ecef = ECEF(self)
        pos_v = np.array(
            (ecef.x - ltp.location.x, ecef.y - ltp.location.y, ecef.z - ltp.location.z)
        )
        ltp_cord = np.matmul(ltp.basis, pos_v)
        x, y, z = ltp_cord[0], ltp_cord[1], ltp_cord[2]

        return LTP(x=x, y=y, z=z, frame=ltp)

    def ltp_to_grandcs(self):
        r"""Returns this vector in the :class:`GRANDCS` array frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # just instantiating a GRANDCS CS to get it's basis and location. x, y, z values does not matter.
        self = copy(self)
        gcs = GRANDCS(x=0, y=0, z=0)
        self.ltp_to_ltp(gcs)

    def ltp_to_ecef(self):
        r"""Returns this vector in the :class:`ECEF` frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # Basis forms a rotational matrix. Transpose is the inverse of rotational matrix (real).
        # Use inverse (transpose) of rotational matrix to convert from GRANDCS to ECEF.
        return ECEF(self)

    def ltp_to_geodetic(self, reference="GEOID"):
        r"""Returns this vector as a geodetic position.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # Convert from GRANDCS to ECEF, then from ECEF to Geodetic.
        ecef = ECEF(self)
        return Geodetic(ecef, reference=reference)


class GRANDCS(LTP):
    """
    Class for the GRANDCS coordinate system (cs). This class instantiate coordinates.

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
        r"""Initialises the GRAND array frame at a given origin.

        An :class:`LTP` with GRAND's conventions applied: the axes are
        north-west-up, so **x runs north**, not east.

        Parameters
        ----------
        arg : Geodetic, ECEF, LTP or GRANDCS, optional
            A position to convert into the array frame.
        x, y, z : float or ndarray, optional
            Components in metres: x north, y west, z up.
        latitude, longitude, height : float or ndarray, optional
            Origin of the frame, if not given through `location`.
        location : Geodetic, ECEF, LTP or GRANDCS, optional
            Origin of the frame.
        obstime : str or datetime, optional
            Date the coordinates refer to.

        Examples
        --------
        .. jupyter-execute::

            import numpy as np
            from grand import Geodetic, GRANDCS

            site = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)
            north = GRANDCS(x=1000.0, y=0.0, z=0.0, location=site)
            print(np.round(np.asarray(Geodetic(north)).ravel(), 6))

        The latitude rises, confirming that x runs north.
        """
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
        r"""Returns this vector in the :class:`ECEF` frame.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # Basis forms a rotational matrix. Transpose is the inverse of rotational matrix (real).
        # Use inverse (transpose) of rotational matrix to convert from GRANDCSCS to ECEF.
        return self.ltp_to_ecef()

    def grandcs_to_geodetic(self, reference="GEOID"):
        r"""Returns this vector as a geodetic position.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # Convert from GRANDCSCS to ECEF, then from ECEF to Geodetic.
        return self.ltp_to_geodetic(reference=reference)

    def grandcs_to_ltp(self, ltp):
        r"""Returns this vector in a local tangent plane, :class:`LTP`.

        Every conversion between a local frame and geodetic passes through
        :class:`ECEF`; see :doc:`/coordinates`.
        """
        # Convert from GRANDCSCS to ECEF, then from ECEF to Geodetic.
        return self.ltp_to_ltp(ltp)


# RK TODO: Rework on this class.
class Rotation(_Rotation):
    r"""A rotation between coordinate frames.

        Extends :class:`scipy.spatial.transform.Rotation` so that a rotation can
        be applied to the frame objects in this module and carry their metadata
        through.
    """
    pass