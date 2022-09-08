"""Topography wrapper for GRAND packages.
"""

from __future__ import annotations

import enum
import os
from pathlib import Path
from typing import Optional, Union, Any
from typing_extensions import Final

import numpy as np
from . import DATADIR
from grand.geo.coordinates import (
    ECEF,
    Geodetic,
    GeodeticRepresentation,
    LTP,
    GRANDCS,
    CartesianRepresentation,
)
from ..libs.turtle import Map as _Map, Stack as _Stack, Stepper as _Stepper
from .. import store
from .._core import ffi, lib

__all__ = [
    "elevation",
    "distance",
    "geoid_undulation",
    "update_data",
    "cachedir",
    "model",
    "Reference",
    "Topography",
]


class Reference(enum.IntEnum):
    """Reference level for topography data"""

    ELLIPSOID = enum.auto()
    GEOID = enum.auto()
    LOCAL = enum.auto()


_CACHEDIR: Final = Path(__file__).parent / "data" / "topography"
"""Location of cached topography data"""


_DEFAULT_MODEL: Final = "SRTMGL1"
"""The default topographic model"""


_default_topography: Optional["Topography"] = None
"""Stack for the topographic data"""

_default_reference: Optional[str] = "GEOID"  # options: 'GEOID', LOCAL', 'ELLIPSOID'
"""Stack for the topographic data"""

_geoid: Optional[_Map] = None
"""Map with geoid undulations"""


def distance(
    position: Any,
    direction: CartesianRepresentation,
    maximum_distance: float = None,
):
    """Get the signed intersection distance with the topography."""
    global _default_topography

    if _default_topography is None:
        _CACHEDIR.mkdir(exist_ok=True)
        _default_topography = Topography(_CACHEDIR)
    return _default_topography.distance(position, direction, maximum_distance)


def elevation(coordinates, reference: Optional[str] = _default_reference):
    """Get the topography elevation, w.r.t. sea level or w.r.t. the ellipsoid."""
    global _default_topography

    if _default_topography is None:
        _CACHEDIR.mkdir(exist_ok=True)
        _default_topography = Topography(_CACHEDIR)
    return _default_topography.elevation(coordinates, reference)


def _get_geoid():
    global _geoid

    if _geoid is None:
        path = os.path.join(DATADIR, "egm96.png")
        _geoid = _Map(path)
    return _geoid


def geoid_undulationX(coordinates):
    """Get the geoid undulation. This function calculates the height of
    the geoid w.r.t the ellipsoid at a given latitude and longitude.
    """
    geoid = _get_geoid()

    # Compute the geodetic coordinates
    geodetic = Geodetic(coordinates)
    z = geoid.elevation(geodetic.longitude, geodetic.latitude)

    return z


def geoid_undulation(coordinates=None, latitude=None, longitude=None):
    """Get the geoid undulation. This function calculates the height of
    the geoid w.r.t the ellipsoid at a given latitude and longitude.
    """
    geoid = _get_geoid()

    # Compute the geodetic coordinates
    # if (not isinstance(latitude, type(None))) and (not isinstance(longitude, type(None))):
    if (latitude is not None) and (longitude is not None):
        pass
        # elif not isinstance(coordinates, type(None)):
    elif coordinates is not None:
        geodetic = Geodetic(coordinates)
        latitude = geodetic.latitude
        longitude = geodetic.longitude
    else:
        raise TypeError(
            "Provide coordinates in known coordinate frames or as latitude and longitude."
        )

    return geoid.elevation(longitude, latitude)


def update_data(coordinates=None, clear: bool = False, radius: float = None):

    """Update the cache of topography data.
    Data are stored in https://github.com/grand-mother/store/releases.
    Locally saved as .../grand/grand/tools/data/topography/*.SRTMGL1.hgt
    """
    if clear:
        for p in _CACHEDIR.glob("**/*.*"):
            p.unlink()

    if coordinates is not None:
        _CACHEDIR.mkdir(exist_ok=True)

        # Compute the bounding box
        if isinstance(coordinates, (ECEF, Geodetic, GeodeticRepresentation, GRANDCS, LTP)):
            pass
        else:
            raise TypeError(
                type(coordinates),
                "Coordinate must be in ECEF, Geodetic, GeodeticRepresentaion, GRAND or LTP.",
            )

        coordinates = Geodetic(coordinates)
        latitude = coordinates.latitude
        longitude = coordinates.longitude
        height = coordinates.height
        # latitude and longitude are stored as ndarray. Find minimum and maximum.
        latitude = [min(latitude), max(latitude)]
        longitude = [min(longitude), max(longitude)]
        height = [min(height), max(height)]

        # Extend by the radius, if any
        if radius is not None:
            for i in range(2):
                # define a local LTP frame at a given latitude and longitude.
                location = Geodetic(latitude=latitude[i], longitude=longitude[i], height=height[i])
                c = LTP(location=location, orientation="ENU", magnetic=False)
                basis = c.basis  # in ECEF frame
                origin = c.location  # in ECEF frame

                # Find the maximum latitude and longitude at radius distance from the LTP origin.
                # 3 points defined at radius distance from the origin, one point on each axis.
                # Max latitude = origin+radius towards N. Min latitude = origin-radius towards N.
                # Max longitude = origin+radius towards E. Min longitude = origin-radius towards E.
                delta = -1 * radius if not i else radius
                ltp_E = np.array([delta, 0, 0])  # delta distance [m] towards E from origin.
                ltp_N = np.array([0, delta, 0])  # delta distance [m] towards N from origin.
                ltp_U = np.array([0, 0, delta])  # delta distance [m] towards U from origin.
                arg = np.column_stack(
                    (ltp_E, ltp_N, ltp_U)
                )  # [[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]]
                # Transform all 3 points from local LTP to ECEF frame.
                ecef = np.matmul(basis.T, arg) + origin
                geod = Geodetic(ecef)
                latitude[i] = min(geod.latitude) if not i else max(geod.latitude)
                longitude[i] = min(geod.longitude) if not i else max(geod.longitude)
                height[i] = min(geod.height) if not i else max(geod.height)

        # Get the corresponding tiles
        longitude = [int(np.floor(lon)) for lon in longitude]
        latitude = [int(np.floor(lat)) for lat in latitude]

        for lat in range(latitude[0], latitude[1] + 1):
            for lon in range(longitude[0], longitude[1] + 1):
                ns = "S" if lat < 0 else "N"
                ew = "W" if lon < 0 else "E"
                lat = -lat if lat < 0 else lat
                lon = -lon if lon < 0 else lon

                basename = f"{ns}{lat:02.0f}{ew}{lon:03.0f}.SRTMGL1.hgt"
                path = _CACHEDIR / basename
                if not path.exists():
                    print("Caching data for", path)
                    try:
                        data = store.get(
                            basename
                        )  # stored in github.com/grand-mother/store/releases.
                    except store.InvalidBLOB:
                        raise ValueError(
                            f"missing data for {basename}"
                        ) from None  # RK: what is this? and why?
                    else:
                        with path.open("wb") as f:
                            f.write(data)

                # ToDo: Add error message if failing to load topography data.

    # Reset the topography proxy
    global _default_topography
    _default_topography = None


def cachedir() -> Path:
    """Get the location of the topography data cache."""
    return _CACHEDIR


def model() -> str:
    """Get the default model for topographic data."""
    return _DEFAULT_MODEL


class Topography:
    """Proxy to topography data."""

    def __init__(self, path: Union[Path, str] = _CACHEDIR) -> None:
        self._stack = _Stack(str(path))
        self._stepper: Optional[_Stepper] = None

    def elevation(
        self,
        coordinates,
        reference: Optional[str] = _default_reference,
    ):
        """Get the topography elevation, w.r.t. sea level, w.r.t the
        ellipsoid or in local coordinates. The default reference is
        w.r.t sea level (GEOID).
        """
        if isinstance(reference, str):
            reference = reference.upper()

            if reference == "LOCAL":
                if not isinstance(coordinates, (LTP, GRANDCS)):
                    raise ValueError("not an LTP or GRANDCS frame")
                elevation = self._local_elevation(coordinates)
            else:
                elevation = self._global_elevation(coordinates, reference)

            if elevation.size == 1:
                elevation = elevation[0]

            return elevation
        else:
            # TODO: what doing if reference is None ?
            raise ValueError

    @staticmethod
    def _as_double_ptr(a):
        a = np.require(a, float, ["CONTIGUOUS", "ALIGNED"])
        return ffi.cast("double *", a.ctypes.data)

    def _local_elevation(self, coordinates):
        """Get the topography elevation in local coordinates, i.e. along the (Oz) axis."""
        # Compute the x and y coordinate in local frame.
        x = coordinates.x
        y = coordinates.y
        if not isinstance(x, np.ndarray):
            x = np.array((x,))
            y = np.array((y,))

        # Return the topography elevation
        n = x.size
        elevation = np.zeros(n)
        origin = coordinates.location
        basis = (
            coordinates.basis.T
        )  # basis in coordinates.py and in lib... are transpose of each other.
        geoid = _get_geoid()._map[0]
        stack = self._stack._stack[0] if self._stack._stack else ffi.NULL

        lib.grand_topography_local_elevation(
            stack,
            geoid,
            self._as_double_ptr(origin),
            self._as_double_ptr(basis),
            self._as_double_ptr(x),
            self._as_double_ptr(y),
            self._as_double_ptr(elevation),
            n,
        )

        return elevation

    def _global_elevation(self, coordinates, reference: str):
        """Get the topography elevation w.r.t. sea level or w.r.t. the
        ellipsoid.
        """

        # Compute the geodetic coordinates
        geodetic = Geodetic(coordinates)
        latitude = geodetic.latitude
        longitude = geodetic.longitude
        if not isinstance(latitude, np.ndarray):
            latitude = np.array((latitude,))
            longitude = np.array((longitude,))

        # Return the topography elevation
        n = latitude.size
        elevation = np.zeros(n)
        if reference == "ELLIPSOID":
            geoid = _get_geoid()._map[0]
        else:
            geoid = ffi.NULL
        stack = self._stack._stack[0] if self._stack._stack else ffi.NULL

        lib.grand_topography_global_elevation(
            stack,
            geoid,
            self._as_double_ptr(latitude),
            self._as_double_ptr(longitude),
            self._as_double_ptr(elevation),
            n,
        )

        return elevation

    def distance(
        self,
        position: Any,
        direction: CartesianRepresentation,
        maximum_distance: float = None,
    ):
        """Get the signed intersection distance with the topography."""
        if self._stepper is None:
            stepper = _Stepper()
            stepper.add(self._stack)
            stepper.geoid = _get_geoid()
            self._stepper = stepper

        position = ECEF(position)
        if isinstance(direction, (CartesianRepresentation, ECEF)):
            # TODO: Convert direction vector given in any known coordinate frame to ECEF frame.
            #       direction must be in ECEF frame for lib.grand_topography_distance()
            pass
        else:
            raise TypeError("Direction must be in CartesianRepresentation in ECEF frame.")

        # Normalize the direction vector. Unit vector is required.
        norm = np.linalg.norm(direction)
        direction = direction / norm

        dn = np.float64(maximum_distance).size if maximum_distance is not None else 1
        n = max(position.x.size, direction.x.size, dn)

        if (
            ((direction.size > 1) and (direction.size < n))
            or ((position.size > 1) and (position.size < n))
            or ((dn > 1) and (dn < n))
        ):
            raise ValueError("incompatible size")

        r = np.empty(3 * n)
        v = np.empty(3 * n)
        d = np.empty(n)

        # r[start:stop:step] -> r[start::step]. Take every step-th value starting from start.
        # l = [0,1,2,3,4,5]. l[::2] -> [0, 2, 4]. l[1::2] -> [1, 3, 5]
        r[::3] = position.x
        r[1::3] = position.y
        r[2::3] = position.z
        v[::3] = direction.x
        v[1::3] = direction.y
        v[2::3] = direction.z
        d[:] = maximum_distance if maximum_distance is not None else 0

        lib.grand_topography_distance(
            self._stepper._stepper[0],
            self._as_double_ptr(r),
            self._as_double_ptr(v),
            self._as_double_ptr(d),
            n,
        )

        if d.size == 1:
            d = d[0]

        return d
