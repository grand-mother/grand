"""GRAND software package
"""

import os
import os.path as osp
from pathlib import Path

from grand.geo import geomagnet, topography
from grand.geo.topography import geoid_undulation, Reference, Topography
from grand.geo import coordinates
from grand.geo.geomagnet import Geomagnet
from grand.geo.coordinates import (
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
import grand.io.protocol as store


def grand_get_path_root_pkg():
    """get the root path of grand git package, ex: /home/user/grand

    @return (string) : root path of grand git package
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
    """
    return osp.join(grand_get_path_root_pkg(), "grand")


GRAND_DATA_PATH = osp.join(grand_get_path_root_pkg(), "data")


def grand_add_path_data(s_file):
    return os.path.join(GRAND_DATA_PATH, s_file)


def grand_add_path_data_model(s_file):
    return os.path.join(GRAND_DATA_PATH, "model", s_file)


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
    "grand_add_path_data",
]
