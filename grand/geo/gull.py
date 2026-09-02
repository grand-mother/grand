"""Wrapper for the GULL C library
"""

from logging import getLogger
import datetime

import numpy

#from . import DATADIR
from .._core import ffi, lib


__all__ = ["LibraryError", "Snapshot"]

logger = getLogger(__name__)


class LibraryError(RuntimeError):
    """A GULL library error"""

    def __init__(self, code):
        """Set a GULL library error

        Parameters
        ----------
        code : int
            The function return code
        """
        self.code = code
        error = ffi.string(lib.grand_error_get())
        message = f"A GULL library error occurred: {error}"

        super().__init__(message)


class Snapshot:
    """Proxy for a GULL snapshot object"""

    def __init__(self, filename="", date="2020-01-01"):
        """Create a snapshot of the geo-magnetic field

        Parameters
        ----------
        filename : str
            The filename with geo-magnetic model info is stored (IGRF13.COF, or WMM2020.COF)
        date : str or datetime.date
            The day at which the snapshot is taken

        Raises
        ------
        LibraryError
            A GULL library error occured, e.g. if the model parameters are not
            valid
        """
        #self._snapshot, self._model, self._date = None, None, None
        self._snapshot, self._date = None, None
        self._workspace = ffi.new("double **")
        self._order, self._altitude = None, None

        # Create the snapshot object
        snapshot = ffi.new("struct gull_snapshot **")
        if isinstance(date, str):
            d = datetime.date.fromisoformat(date)
        else:
            d = date

        #path = ffi.new("char []", f"{DATADIR}/gull/{model}.COF".encode())
        #path = ffi.new("char []", f"{DATADIR}/{model}.COF".encode())
        path  = ffi.new("char []", filename.encode())
        line = ffi.new("int *")

        r = lib.gull_snapshot_create(snapshot, path, d.day, d.month, d.year)
        if r != 0:
            raise LibraryError(r)
        self._snapshot = snapshot
        #self._model, self._date = model, d
        self._date = d

        # Get the meta-data
        order = ffi.new("int *")
        altitude_min = ffi.new("double *")
        altitude_max = ffi.new("double *")
        lib.gull_snapshot_info(self._snapshot[0], order, altitude_min, altitude_max)
        self._order = int(order[0])
        self._altitude = tuple(map(float, (altitude_min[0], altitude_max[0])))

    def __del__(self):
        r"""Releases the underlying GULL snapshot.

        """
        logger.debug("__del__ Snapshot ")
        try:
            if self._snapshot is None:
                return
        except AttributeError:
            return

        lib.gull_snapshot_destroy(self._snapshot)
        lib.gull_snapshot_destroy(ffi.cast("struct gull_snapshot **", self._workspace))
        self._snapshot = None

    def __call__(self, latitude, longitude, altitude=None):
        """Get the magnetic field at a given Earth location

        Parameters
        ----------
        latitude : float or ndarray
            Degrees north.
        longitude : float or ndarray
            Degrees east.
        altitude : float or ndarray, optional
            Metres above the ellipsoid.

        Returns
        -------
        ndarray
            Field components, in tesla.
        """

        def regularize(a):
            r"""Returns `a` as a contiguous array of the type the GULL bindings expect.

            Parameters
            ----------
            a : array_like
                Value to convert.

            Returns
            -------
            ndarray
                Contiguous ``float64`` array.
            """
            a = numpy.asanyarray(a)
            return numpy.require(a, float, ["CONTIGUOUS", "ALIGNED"])

        latitude, longitude = map(regularize, (latitude, longitude))
        if latitude.size != longitude.size:
            raise ValueError("latitude and longitude must have the same size")

        if altitude is None:
            altitude = numpy.zeros_like(latitude)
        else:
            altitude = regularize(altitude)
            if latitude.size != altitude.size:
                raise ValueError("latitude and altitude must have the same size")

        if latitude.size == 1:
            field = numpy.zeros(3)
        else:
            field = numpy.zeros((latitude.size, 3))

        r = lib.gull_snapshot_field_v(
            self._snapshot[0],
            ffi.cast("double *", latitude.ctypes.data),
            ffi.cast("double *", longitude.ctypes.data),
            ffi.cast("double *", altitude.ctypes.data),
            ffi.cast("double *", field.ctypes.data),
            latitude.size,
            self._workspace,
        )
        if r != 0:
            raise LibraryError(r)

        return field

    @property
    def altitude(self):
        """The altitude range of the snapshot

        Returns
        -------
        tuple of float
            Altitude range the snapshot is valid over, in metres.
        """
        return self._altitude

    @property
    def date(self):
        """The date of the snapshot

        Returns
        -------
        datetime.date
            Date the field model was evaluated for.
        """
        return self._date

    #@property
    #def model(self):
    #    """The world magnetic model"""
    #    return self._model

    @property
    def order(self):
        """The approximation order of the model

        Returns
        -------
        int
            Maximum spherical-harmonic order of the model.
        """
        return self._order
