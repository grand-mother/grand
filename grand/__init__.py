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


import grand.dataio.protocol as store


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




