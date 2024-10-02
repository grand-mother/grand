"""GRAND software package
"""

import os
import os.path as osp
from pathlib import Path


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
import grand.dataio.protocol as store
from grand.sim import efield2voltage
from grand.sim.efield2voltage import Efield2Voltage
from grand.sim.detector.antenna_model import tabulated_antenna_model, AntennaModel
from grand.sim.detector.process_ant import AntennaProcessing
from grand.sim.detector.rf_chain2 import RFChain
from grand.sim.noise.galaxy import galactic_noise
from grand.sim.shower.gen_shower import ShowerEvent
from grand.sim.shower.pdg import ParticleCode


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
    "galactic_noise",
    "ShowerEvent",
    "ParticleCode",
]




