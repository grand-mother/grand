"""Topography wrapper for GRAND packages.
"""

from __future__ import annotations

import enum
import warnings
from pathlib import Path
from typing import Optional, Union, Any
from typing_extensions import Final
import numpy as np

from grand.basis import validate as _validate

from grand import grand_get_path_root_pkg

from . import DATADIR
from grand.geo.coordinates import (
    ECEF,
    Geodetic,
    GeodeticRepresentation,
    LTP,
    GRANDCS,
    CartesianRepresentation,
)

from .turtle import Map as _Map, Stack as _Stack, Stepper as _Stepper
import grand.dataio.protocol as store
try:
    from .._core import ffi, lib
except ImportError as _error:          # (#280)
    from grand import CORE_MISSING
    raise ImportError(CORE_MISSING) from _error

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


DATADIR: Final = Path(grand_get_path_root_pkg()) / "data" / "topography"
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
    frame: Any = None,
):
    """Get the signed intersection distance with the topography.

    Parameters
    ----------
    position : Geodetic, ECEF, LTP or GRANDCS
        Starting point.
    direction : CartesianRepresentation or ECEF
        Direction to travel in, in **ECEF** unless `frame` says otherwise.
    maximum_distance : float, optional
        Give up beyond this distance, in metres.
    frame : LTP, GRANDCS or "ENU", optional
        Frame `direction` is given in: the axes of an `LTP` or `GRANDCS`, or
        ``"ENU"`` for east, north and up at the (single) starting point.  By
        default it is ECEF.

    Returns
    -------
    float or ndarray
        Distance to the ground, in metres, or NaN if it was not reached.
    """
    global _default_topography

    if _default_topography is None:
        DATADIR.mkdir(exist_ok=True)
        _default_topography = Topography(DATADIR)
    return _default_topography.distance(position, direction, maximum_distance, frame=frame)


def _direction_to_ecef(direction, frame, position):
    r"""Rotates `direction`, given in `frame`, to ECEF (see `Topography.distance`)."""
    if isinstance(frame, str):
        if frame.upper() != "ENU":
            raise ValueError("GRANDlib: topography.distance: frame must be an LTP, a GRANDCS "
                             "or \"ENU\", got %r" % frame)
        if position.x.size != 1:
            raise ValueError("GRANDlib: topography.distance: frame=\"ENU\" needs a single "
                             "starting point; pass an LTP as the frame for several")
        start = Geodetic(position)
        frame = LTP(location=Geodetic(latitude=float(np.ravel(start.latitude)[0]),
                                      longitude=float(np.ravel(start.longitude)[0]),
                                      height=float(np.ravel(start.height)[0])),
                    orientation="ENU", magnetic=False)
    basis = getattr(frame, "basis", None)
    if basis is None:
        raise ValueError("GRANDlib: topography.distance: frame must be an LTP, a GRANDCS "
                         "or \"ENU\", got %s" % type(frame).__name__)
    local = np.vstack([np.ravel(direction.x), np.ravel(direction.y), np.ravel(direction.z)])
    ecef = np.matmul(np.asarray(basis).T, local)
    return CartesianRepresentation(x=ecef[0], y=ecef[1], z=ecef[2])


def elevation(coordinates, reference: Optional[str] = _default_reference):
    """Get the topography elevation, w.r.t. sea level or w.r.t. the ellipsoid.

    Parameters
    ----------
    coordinates : Geodetic, ECEF, LTP or GRANDCS
        Position or positions to evaluate at.
    reference : str or Reference, optional
        Surface the height is measured from: the ellipsoid or the geoid.

    Returns
    -------
    float or ndarray
        Ground elevation, in metres.
    """
    global _default_topography

    if _default_topography is None:
        DATADIR.mkdir(exist_ok=True)
        _default_topography = Topography(DATADIR)
    return _default_topography.elevation(coordinates, reference)


def _finite_points(a, b, frame_finite, where):
    r"""Returns the mask of the points libturtle can be given (issue #262).

    Parameters
    ----------
    a, b : ndarray
        The two horizontal coordinates of each point.
    frame_finite : bool
        Whether the frame (origin and basis) the points are given in is finite.
    where : str
        The calling function, for the warning.

    Returns
    -------
    ndarray of bool
        True where both coordinates, and the frame, are finite.
    """
    finite = np.ravel(np.isfinite(a) & np.isfinite(b)) & bool(frame_finite)
    if not finite.all():
        warnings.warn(_validate.message(where, "%d of %d points have a non-finite position; "
                                        "their elevation is NaN" % ((~finite).sum(), finite.size)),
                      _validate.GRANDlibWarning, stacklevel=4)
    return finite


def _fill_elevation(n, finite, values):
    r"""Puts the computed elevations back among the skipped points.

    A point with a finite position that comes back NaN lies outside every
    loaded tile. That used to be silent (#280): the warning says how many and
    how to get the tiles.

    Parameters
    ----------
    n : int
        Total number of points.
    finite : ndarray of bool
        Which points were computed.
    values : ndarray
        Elevations of the computed points.

    Returns
    -------
    ndarray
        The `n` elevations, NaN where none could be computed.
    """
    missing = int(np.isnan(values).sum())
    if missing:
        warnings.warn(_validate.message(
            "Topography.elevation", "%d of %d points are outside the loaded topography tiles; "
            "their elevation is NaN. Download the tiles around them with "
            "grand.topography.update_data(coordinates, radius=...)"
            % (missing, values.size)), _validate.GRANDlibWarning, stacklevel=4)
    elevation = np.full(n, np.nan)
    elevation[finite] = values
    return elevation


def _get_geoid():
    r"""Returns the geoid map, loading it on first use.

    Returns
    -------
    turtle.Map
        The EGM96 undulation map shipped in ``data/``.
    """
    global _geoid

    if _geoid is None:
        # path = os.path.join(DATADIR, "egm96.png")
        path = Path(grand_get_path_root_pkg()) / "data" / "egm96.png"
        _geoid = _Map(path)
    return _geoid


def geoid_undulation(coordinates=None, latitude=None, longitude=None):
    """Get the geoid undulation: the height of the geoid above the ellipsoid.

    Same signature and values as :func:`grand.geo.coordinates.geoid_undulation`.

    Parameters
    ----------
    coordinates : Geodetic, ECEF, LTP, GRANDCS or float, optional
        Position or positions to evaluate at; or, as a number, the latitude,
        with the longitude as the second argument.
    latitude : float or ndarray, optional
        Degrees north, instead of `coordinates`.
    longitude : float or ndarray, optional
        Degrees east, instead of `coordinates`.

    Returns
    -------
    float or ndarray
        Height of the geoid above the ellipsoid, in metres.

    A missing angle gives NaN, with a warning.

    Examples
    --------
    Negative here: at Dunhuang the geoid lies below the ellipsoid, so a point
    1200 m above the ellipsoid is about 1261 m above sea level.

    .. jupyter-execute::

        from grand.geo.topography import geoid_undulation

        print("%.2f m" % geoid_undulation(latitude=40.98, longitude=93.95))
    """
    from grand.geo.coordinates import _latitude_longitude
    latitude, longitude = _latitude_longitude(coordinates, latitude, longitude, "geoid_undulation")
    geoid = _get_geoid()
    # The map spans longitudes 0 to 360: negative ones gave NaN (#251)
    return geoid.elevation(np.mod(np.asarray(longitude, dtype=float), 360.0), latitude)


def update_data(coordinates=None, clear: bool = False, radius: float = None):
    """
    Updates the cache of topography data.

    Tiles are published at https://github.com/grand-mother/store/releases and
    saved locally as ``grand/data/topography/*.hgt``, one file per one-degree
    square.  They are not in version control.

    Parameters
    ----------
    coordinates : Geodetic, ECEF, LTP or GRANDCS
        Position or positions to evaluate at.
    clear : bool, optional
        Remove the cached tiles first.
    radius : float, optional
        Fetch tiles within this radius, in metres.
    """
    if clear:
        for p in DATADIR.glob("**/*.*"):
            p.unlink()

    if coordinates is not None:
        DATADIR.mkdir(exist_ok=True)

        # Compute the bounding box
        if isinstance(coordinates, (ECEF, Geodetic, GeodeticRepresentation, GRANDCS, LTP)):
            pass
        else:
            raise TypeError(
                type(coordinates),
                "Coordinate must be in ECEF, Geodetic, GeodeticRepresentaion, GRANDCS or LTP.",
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

                base = f"{ns}{lat:02.0f}{ew}{lon:03.0f}"
                basename = f"{ns}{lat:02.0f}{ew}{lon:03.0f}.SRTMGL1.hgt" # .gz will be added in protocol.py
                print("topography:", ns, lat, ew, lon, basename)
                # path = DATADIR / basename
                path = DATADIR / (base + ".hgt")
                if not path.exists():
                    print("Caching data for", path)
                    try:
                        data = store.get(
                            basename
                        )  # stored in github.com/grand-mother/store/releases.
                    except store.InvalidBLOB:
                        raise ValueError(
                            f"missing data in GRAND repository for {basename}. Download it from https://search.earthdata.nasa.gov/search/granules?p=C1000000240-LPDAAC_ECS&pg[0][v]=f&pg[0][gsk]=-start_date&tl=1680158515.135!3!!"
                        ) from None  # RK: what is this? and why?
                    else:
                        with path.open("wb") as f:
                            f.write(data)

                # ToDo: Add error message if failing to load topography data.

    # Reset the topography proxy
    global _default_topography
    _default_topography = None

def cachedir() -> Path:
    """Get the location of the topography data cache.

    Returns
    -------
    Path
        Directory the downloaded elevation tiles are cached in.
    """
    return DATADIR


def datadir() -> Path:
    """Get the location of the topography data cache.

    Returns
    -------
    Path
        Directory holding the topography data.
    """
    return DATADIR


def model() -> str:
    """Get the default model for topographic data.

    Returns
    -------
    str
        Name of the elevation model in use, such as SRTM.
    """
    return _DEFAULT_MODEL


class Topography:
    """Proxy to topography data."""

    def __init__(self, path: Union[Path, str] = DATADIR) -> None:
        # self._stack = _Stack(str(path))
        r"""Opens a topography dataset.

        Parameters
        ----------
        path : str
            Directory holding the elevation tiles.
        """
        self._stack = _Stack(path)
        self._stepper: Optional[_Stepper] = None

    def elevation(
        self,
        coordinates,
        reference: Optional[str] = _default_reference,
    ):
        """Get the topography elevation, w.r.t. sea level, w.r.t the
        ellipsoid or in local coordinates. The default reference is
        w.r.t sea level (GEOID).

        Parameters
        ----------
        coordinates : Geodetic, ECEF, LTP or GRANDCS
            Position or positions to evaluate at.
        reference : str or Reference, optional
            Surface the height is measured from: the ellipsoid or the geoid.

        Returns
        -------
        float or ndarray
            Ground elevation, in metres.
        """
        if isinstance(reference, str):
            reference = reference.upper()
            # "SEA" (sea level) has always been accepted and means the geoid
            _validate.one_of(reference, ("GEOID", "SEA", "ELLIPSOID", "LOCAL"), "reference",
                             "Topography.elevation")

            if reference == "LOCAL":
                if not isinstance(coordinates, (LTP, GRANDCS)):
                    raise TypeError(_validate.message(
                        "Topography.elevation", "reference='LOCAL' needs coordinates in an LTP or "
                        "GRANDCS frame, got %s" % type(coordinates).__name__))
                elevation = self._local_elevation(coordinates)
            else:
                elevation = self._global_elevation(coordinates, reference)

            if elevation.size == 1:
                elevation = elevation[0]

            return elevation
        else:
            raise TypeError(_validate.message(
                "Topography.elevation", "'reference' must be 'GEOID' (or 'SEA'), 'ELLIPSOID' or 'LOCAL', "
                "got %r" % (reference,)))

    @staticmethod
    def _as_double_ptr(a):
        r"""Returns `a` as a C pointer to double, for the TURTLE bindings.

        Parameters
        ----------
        a : ndarray
            Array to pass through; converted to ``float64`` if needed.

        Returns
        -------
        cffi pointer
            Pointer to the array's data.
        """
        a = np.require(a, float, ["CONTIGUOUS", "ALIGNED"])
        return ffi.cast("double *", a.ctypes.data)

    def _local_elevation(self, coordinates):
        """Get the topography elevation in local coordinates, i.e. along the (Oz) axis.

        Parameters
        ----------
        coordinates : Geodetic, ECEF, LTP or GRANDCS
            Position or positions to evaluate at.

        Returns
        -------
        ndarray
            Elevation from the local tiles, in metres.
        """
        # Compute the x and y coordinate in local frame.
        x = coordinates.x
        y = coordinates.y
        if not isinstance(x, np.ndarray):
            x = np.array((x,))
            y = np.array((y,))

        # Return the topography elevation
        n = x.size
        origin = np.ascontiguousarray(coordinates.location, dtype=float)
        basis = np.ascontiguousarray(
            coordinates.basis.T, dtype=float
        )  # basis in coordinates.py and in lib... are transpose of each other.
        # libturtle crashes on a NaN coordinate (issue #262): only finite
        # points are passed to it, the others get NaN with a warning.
        finite = _finite_points(x, y, np.all(np.isfinite(origin)) and np.all(np.isfinite(basis)),
                                "Topography.elevation")
        x, y = (np.ascontiguousarray(np.ravel(v)[finite], dtype=float) for v in (x, y))
        values = np.zeros(x.size)
        geoid = _get_geoid()._map[0]
        stack = self._stack._stack[0] if self._stack._stack else ffi.NULL

        lib.grand_topography_local_elevation(
            stack,
            geoid,
            self._as_double_ptr(origin),
            self._as_double_ptr(basis),
            self._as_double_ptr(x),
            self._as_double_ptr(y),
            self._as_double_ptr(values),
            x.size,
        )

        return _fill_elevation(n, finite, values)

    def _global_elevation(self, coordinates, reference: str):
        """Get the topography elevation w.r.t. sea level or w.r.t. the
        ellipsoid.

        Parameters
        ----------
        coordinates : Geodetic, ECEF, LTP or GRANDCS
            Position or positions to evaluate at.
        reference : str or Reference, optional
            Surface the height is measured from: the ellipsoid or the geoid.

        Returns
        -------
        ndarray
            Elevation from the global model, in metres.
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
        # libturtle crashes on a NaN coordinate (issue #262): only finite
        # points are passed to it, the others get NaN with a warning.
        finite = _finite_points(latitude, longitude, True, "Topography.elevation")
        latitude, longitude = (np.ascontiguousarray(np.ravel(v)[finite], dtype=float)
                               for v in (latitude, longitude))
        values = np.zeros(latitude.size)
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
            self._as_double_ptr(values),
            latitude.size,
        )

        return _fill_elevation(n, finite, values)

    def distance(
        self,
        position: Any,
        direction: CartesianRepresentation,
        maximum_distance: float = None,
        frame: Any = None,
    ):
        """Get the signed intersection distance with the topography.

        Parameters
        ----------
        position : Geodetic, ECEF, LTP or GRANDCS
            Starting point.
        direction : CartesianRepresentation or ECEF
            Direction to travel in, in **ECEF** unless `frame` says otherwise.
            A local (east, north, up) vector passed without `frame` is read as
            ECEF and gives a wrong distance (#210).
        maximum_distance : float, optional
            Give up beyond this distance, in metres.
        frame : LTP, GRANDCS or "ENU", optional
            Frame `direction` is given in: the axes of an `LTP` or `GRANDCS`,
            or ``"ENU"`` for east, north and up at the (single) starting
            point.  By default it is ECEF.

        Returns
        -------
        float or ndarray
            Distance to the ground, in metres.
        """
        if self._stepper is None:
            stepper = _Stepper()
            stepper.add(self._stack)
            stepper.geoid = _get_geoid()
            self._stepper = stepper

        position = ECEF(position)
        if not isinstance(direction, (CartesianRepresentation, ECEF)):
            raise TypeError("Direction must be in CartesianRepresentation in ECEF frame.")
        # TURTLE needs an ECEF direction (#210)
        if frame is not None:
            direction = _direction_to_ecef(direction, frame, position)

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
