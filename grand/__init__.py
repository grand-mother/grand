"""GRAND software package
"""

import os
import os.path as osp
from pathlib import Path


def grand_get_path_root_pkg():
    """get the root path of grand git package, ex: /home/user/grand

    @return (string) : root path of grand git package

    Returns
    -------
    str
        Root directory of the git checkout.
    """
    root = os.getenv("GRAND_ROOT")
    if not root:
        l_sep = osp.sep
        full = __file__.split(l_sep)
        root = l_sep.join(full[:-2])
    return root


def grand_get_path_grandlib():
    """get root path of grand source, ex: /home/user/grand/grand

    @return (string) : root path of grand source

    Returns
    -------
    str
        Directory holding the ``grand`` package.
    """
    return osp.join(grand_get_path_root_pkg(), "grand")


GRAND_DATA_PATH = osp.join(grand_get_path_root_pkg(), "data")


def grand_add_path_data(s_file):
    r"""Returns the absolute path of a file in the package's data directory.

    Parameters
    ----------
    s_file : str
        Path relative to ``data/``.

    Returns
    -------
    str
    Absolute path.
    """
    return os.path.join(GRAND_DATA_PATH, s_file)


# The public names below are loaded on first use, not at import.
#
# They used to be imported here eagerly, so *any* import from the package --
# ``import grand.dataio`` to read a file -- loaded 13 grand.sim and 6 grand.geo
# modules and the compiled C core (TURTLE, GULL).  Reading GRAND files
# therefore required building C extensions whose only purpose is coordinate
# and topography physics, and failed with "No module named 'grand._core'"
# when they were absent.  The ``grandio_light`` branch proposed deleting the
# physics from the repository to get an I/O-only package; loading lazily gets
# the same result without deleting anything.  See
# resources/dev/dev-next/DECISIONS.md.
#
# Behaviour for users is unchanged: ``grand.Geodetic``,
# ``from grand import Efield2Voltage`` and ``from grand import *`` all work,
# importing the module that defines the name at that moment (PEP 562).
_LAZY = {
    # name: (module, attribute or None for the module itself)
    "geomagnet": ("grand.geo.geomagnet", None),
    "topography": ("grand.geo.topography", None),
    "coordinates": ("grand.geo.coordinates", None),
    "geoid_undulation": ("grand.geo.topography", "geoid_undulation"),
    "Reference": ("grand.geo.topography", "Reference"),
    "Topography": ("grand.geo.topography", "Topography"),
    "Geomagnet": ("grand.geo.geomagnet", "Geomagnet"),
    "Coordinates": ("grand.geo.coordinates", "Coordinates"),
    "CartesianRepresentation": ("grand.geo.coordinates", "CartesianRepresentation"),
    "SphericalRepresentation": ("grand.geo.coordinates", "SphericalRepresentation"),
    "GeodeticRepresentation": ("grand.geo.coordinates", "GeodeticRepresentation"),
    "Geodetic": ("grand.geo.coordinates", "Geodetic"),
    "GRANDCS": ("grand.geo.coordinates", "GRANDCS"),
    "LTP": ("grand.geo.coordinates", "LTP"),
    "ECEF": ("grand.geo.coordinates", "ECEF"),
    "HorizontalVector": ("grand.geo.coordinates", "HorizontalVector"),
    "Horizontal": ("grand.geo.coordinates", "Horizontal"),
    "HorizontalRepresentation": ("grand.geo.coordinates", "HorizontalRepresentation"),
    "Rotation": ("grand.geo.coordinates", "Rotation"),
    "store": ("grand.dataio.protocol", None),
    "efield2voltage": ("grand.sim.efield2voltage", None),
    "Efield2Voltage": ("grand.sim.efield2voltage", "Efield2Voltage"),
    "tabulated_antenna_model": ("grand.sim.detector.antenna_model", "tabulated_antenna_model"),
    "AntennaModel": ("grand.sim.detector.antenna_model", "AntennaModel"),
    "AntennaProcessing": ("grand.sim.detector.process_ant", "AntennaProcessing"),
    "RFChain": ("grand.sim.detector.rf_chain", "RFChain"),
    # Listed in __all__ since before this change but never actually imported,
    # so ``from grand import *`` raised AttributeError on it.
    "adc": ("grand.sim.detector.adc", None),
    "ADC": ("grand.sim.detector.adc", "ADC"),
    "galactic_noise": ("grand.sim.noise.galaxy", "galactic_noise"),
    "ShowerEvent": ("grand.sim.shower.gen_shower", "ShowerEvent"),
    "ParticleCode": ("grand.sim.shower.pdg", "ParticleCode"),
}


def __getattr__(name):
    r"""Loads a public name, or a subpackage, the first time it is asked for.

    Parameters
    ----------
    name : str
        The attribute requested from the ``grand`` package.

    Returns
    -------
    object
        The module or object that name refers to.  It is also stored on the
        package, so this runs once per name.
    """
    import importlib

    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        module = importlib.import_module(module_name)
        value = module if attribute is None else getattr(module, attribute)
    else:
        # ``grand.sim`` and the like after a bare ``import grand``: the
        # subpackages used to be loaded as a side effect of the eager imports.
        try:
            value = importlib.import_module("grand." + name)
        except ModuleNotFoundError as error:
            if error.name != "grand." + name:
                raise
            raise AttributeError("module 'grand' has no attribute %r" % name) from None
    globals()[name] = value
    return value


def __dir__():
    r"""Lists the lazily loaded names alongside the ones already present.

    Returns
    -------
    list of str
        Every public name of the package.
    """
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "GRAND_DATA_PATH",
    "grand_add_path_data",
    "geomagnet", "Geomagnet", "topography", "Topography",
    "geoid_undulation", "Reference",
    "Coordinates", "CartesianRepresentation", "SphericalRepresentation", "GeodeticRepresentation",
    "Geodetic", "ECEF", "LTP", "GRANDCS", "Rotation", 
    "store",
    "efield2voltage", "Efield2Voltage",
    "tabulated_antenna_model", "AntennaModel", "AntennaProcessing",
    "RFChain",
    "adc", "ADC",
    "galactic_noise",
    "ShowerEvent",
    "ParticleCode",
]




