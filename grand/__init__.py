"""GRAND software package
"""

import os
import os.path as osp
from pathlib import Path

from .tools import geomagnet, topography
from .tools.topography import geoid_undulation, Reference, Topography
from .tools.geomagnet import Geomagnet
from .tools import coordinates
from .tools.coordinates import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
    Geodetic,
    GRANDCS,
    LTP,
    ECEF,
    HorizontalVector,
    Horizontal,
    HorizontalRepresentation,
    Rotation,
)
from . import store


def get_root_grand_git():
    """get the root path of grand git package, ex: /home/user/grand

    @return (string) : root path of grand git package
    """
    root = os.getenv("GRAND_ROOT")
    if not root:
        l_sep = osp.sep
        full = __file__.split(l_sep)
        root = l_sep.join(full[:-2])
    return root


def get_root_grand_src():
    """get root path of grand source, ex: /home/user/grand/grand

    @return (string) : root path of grand source
    """
    return osp.join(get_root_grand_git(), "grand")


OPTS_DATA_PATH = osp.join(get_root_grand_src(), "tools/data/huge")

if os.path.isfile(osp.join(OPTS_DATA_PATH, "use_instead_HOME.grand")):
    GRAND_DATA_PATH = OPTS_DATA_PATH
else:
    GRAND_DATA_PATH = osp.join(Path.home(), ".grand")


__all__ = [
    "geomagnet",
    "store",
    "topography",
    "ECEF",
    "Geodetic",
    "GeodeticRepresentation",
    "GRANDCS",
    "coordinates",
    "LTP",
    "SphericalRepresentation",
    "CartesianRepresentation",
    "Rotation",
    "GRAND_DATA_PATH",
]
