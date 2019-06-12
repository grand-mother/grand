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

"""Encapsulation of the GULL C library
"""

import ctypes
import datetime
import os
import shutil
import subprocess

import numpy

from . import DATADIR, LIBDIR, SRCDIR
from .tools import Meta, Temporary, define

__all__ = ["LIBNAME", "LIBPATH", "LIBHASH", "LibraryError", "Snapshot",
           "strerror"]


LIBNAME = "libgull.so"
"""The OS specific name of the GULL library object"""


LIBPATH = os.path.join(LIBDIR, LIBNAME)
"""The full path to the GULL library object"""


LIBHASH = "91ed20fc52c35a8ae9d32416dd7d0249100aad6f"
"""The git hash of the library"""


def _install():
    """Install the GULL library to the top package location"""

    # Check for an existing install
    meta = Meta("gull")
    if meta["LIBHASH"] == LIBHASH:
        return

    def system(command):
        subprocess.run(command, check=True, shell=True)

    # Install the library with its vectorization binding
    with Temporary("https://github.com/niess/gull", LIBHASH) as _:
        # Extend the source with vectorization
        target = f"src/gull.c"
        system(f"cat {target} {SRCDIR}/gull.c > tmp.c")
        system(f"mv tmp.c {target}")

        # Build the library
        system("make")

        # Copy back the library
        if not os.path.exists(LIBDIR):
            os.makedirs(LIBDIR)
        src = os.path.join("lib", "libgull.so")
        shutil.copy(src, LIBPATH)

        # Copy the data files
        dstdir = os.path.join(DATADIR, "gull")
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        for fname in ("IGRF12.COF", "WMM2015.COF"):
            dst = os.path.join(dstdir, fname)
            if not os.path.exists(dst):
                shutil.copy(os.path.join("share", "data", fname), dst)

    # Dump the meta data
    meta["LIBHASH"] = LIBHASH
    meta.update()


# Install the library on import, if needed
_install()


def strerror(code):
    """Convert a GULL library return code to a string

    Parameters
    ----------
    code : int
        The function return code

    Returns
    -------
    str
        A string describing the error type
    """

    r = ["RETURN_SUCCESS", "RETURN_DOMAIN_ERROR", "RETURN_FORMAT_ERROR",
         "RETURN_MEMORY_ERROR", "RETURN_MISSING_DATA", "RETURN_PATH_ERROR"]
    return r[code]


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
        message = f"A GULL library error occurred: {strerror(code)}"

        super().__init__(message)


_lib = ctypes.cdll.LoadLibrary(LIBPATH)
"""Proxy for the GULL library"""


@define (_lib.gull_snapshot_create,
         arguments = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_int)),
         result = ctypes.c_int,
         exception = LibraryError)
def _snapshot_create(snapshot, path, day, month, year, line):
    """Create a new snapshot object"""
    pass


@define (_lib.gull_snapshot_destroy,
         arguments=(ctypes.POINTER(ctypes.c_void_p),))
def _snapshot_destroy(snapshot):
    """Destroy a snapshot object"""
    pass

_CST_DBL_P = numpy.ctypeslib.ndpointer(float, flags="aligned, contiguous")
_DBL_P = numpy.ctypeslib.ndpointer(float,
    flags="aligned, contiguous, writeable")

@define (_lib.gull_snapshot_field_v,
         arguments = (ctypes.c_void_p, _CST_DBL_P, _CST_DBL_P, _CST_DBL_P,
                      _DBL_P, numpy.ctypeslib.c_intp,
                      ctypes.POINTER(ctypes.c_void_p)),
         result = ctypes.c_int,
         exception = LibraryError)
def _snapshot_field(snapshot, latitude, longitude, altitude, field, size,
                    workspace):
    """Get a magnetic field value from a snapshot"""
    pass


@define (_lib.gull_snapshot_info,
         arguments=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double)))
def _snapshot_info(snapshot):
    """Get some extra info about the snapshot data"""
    pass


class Snapshot:
    """Proxy for a GULL snapshot object"""

    def __init__(self, model="IGRF12", date="2019-01-01"):
        """Create a snapshot of the geo-magnetic field

        Parameters
        ----------
        model : str
            The geo-magnetic model to use (IGRF12, or WMM2015)
        date : str or datetime.date
            The day at which the snapshot is taken

        Raises
        ------
        LibraryError
            A GULL library error occured, e.g. if the model parameters are not
            valid
        """
        self._snapshot, self._model, self._date = None, None, None
        self._workspace = ctypes.c_void_p(0)
        self._order, self._altitude = None, None

        # Create the snapshot object
        snapshot = ctypes.c_void_p(None)
        if isinstance(date, str):
            d = datetime.date.fromisoformat(date)
        else:
            d = date
        day, month, year = map(ctypes.c_int, (d.day, d.month, d.year))

        path = f"{DATADIR}/gull/{model}.COF".encode("ascii")
        line = ctypes.c_int()

        if (_snapshot_create(ctypes.byref(snapshot), path, day, month, year,
                             ctypes.byref(line)) != 0):
            return
        self._snapshot = snapshot
        self._model, self._date = model, d

        # Get the meta-data
        order = ctypes.c_int()
        altitude_min = ctypes.c_double()
        altitude_max = ctypes.c_double()
        _snapshot_info(self._snapshot, ctypes.byref(order),
                       ctypes.byref(altitude_min), ctypes.byref(altitude_max))
        self._order = order.value
        self._altitude = (altitude_min.value, altitude_max.value)


    def __del__(self):
        try:
            if self._snapshot is None:
                return
        except AttributeError:
            return

        _snapshot_destroy(ctypes.byref(self._snapshot))
        _snapshot_destroy(ctypes.byref(self._workspace))
        self._snapshot = None


    def __call__(self, latitude, longitude, altitude=None):
        """Get the magnetic field at a given Earth location"""
        def regularize(a):
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
                raise ValueError(
                    "latitude and altitude must have the same size")

        if latitude.size == 1:
            field = numpy.zeros(3)
        else:
            field = numpy.zeros((latitude.size, 3))

        if _snapshot_field(self._snapshot, latitude, longitude, altitude,
                           field, latitude.size, ctypes.byref(self._workspace)):
            return
        else:
            return field


    @property
    def altitude(self):
        """The altitude range of the snapshot"""
        return self._altitude


    @property
    def date(self):
        """The date of the snapshot"""
        return self._date


    @property
    def model(self):
        """The world magnetic model"""
        return self._model


    @property
    def order(self):
        """The approximation order of the model"""
        return self._order
