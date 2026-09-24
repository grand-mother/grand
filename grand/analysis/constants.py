"""Physical constants and site defaults used by the reconstruction.

The geomagnetic field at the GP13 DAQ position (``B``, and ``Bvec`` and
``Bn`` with ``x`` along magnetic north) is evaluated on first use, not at
import.  Evaluating it needs the compiled geometry core and the
geomagnetic-model data, so doing it at import made ``import
grand.analysis`` need both -- and broke the documentation build, which
stands a mock in for the core.
"""

from grand import Geodetic, Geomagnet
import numpy as np

c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
n_atm = 1.000136

groundAltitude = 1231


def _geomagnetic():
    r"""Returns the DAQ position and the geomagnetic field there.

    Returns
    -------
    dict
        ``coord_daq``, ``B``, ``Bvec`` and ``Bn``, computed exactly as they
        were when this module evaluated them at import.
    """
    coord_daq = Geodetic(latitude=40.99746387, longitude=93.94868871, height=0)
    B = Geomagnet(location=coord_daq)  # default model and obstime is used. Watch out X = EW!
    # Bvec = [B.field.y[0], -B.field.x[0], B.field.z[0]]  # Bvec_y < 0 because B_g pointing west by 0.6°
    Bvec = [B.field.y[0], 0, B.field.z[0]]  # x = magnetic North!
    Bn = Bvec/np.linalg.norm(Bvec)
    return {"coord_daq": coord_daq, "B": B, "Bvec": Bvec, "Bn": Bn}


def __getattr__(name):
    r"""Computes the geomagnetic constants the first time one is read.

    Parameters
    ----------
    name : str
        The attribute requested.

    Returns
    -------
    object
        ``coord_daq``, ``B``, ``Bvec`` or ``Bn``.  All four are computed
        together and cached on the module.
    """
    if name in ("coord_daq", "B", "Bvec", "Bn"):
        values = _geomagnetic()
        globals().update(values)
        return values[name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))

