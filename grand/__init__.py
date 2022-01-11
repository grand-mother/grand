"""GRAND software package
"""

import os.path as osp
from pathlib import Path


from .tools import geomagnet, topography
from .tools.topography import geoid_undulation, Reference, Topography

# RK
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
    "Logger",
    "LTP",
    "SphericalRepresentation",
    "CartesianRepresentation",
    "Rotation",
    "GRAND_DATA_PATH",
]
