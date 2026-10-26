"""GRAND software package
"""

import os
import os.path as osp
from pathlib import Path

# RK: delete this function. Use 'path_grandlib_root'.
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

# RK: delete this function. Use 'path_grandlib'.
def grand_get_path_grandlib():
    """get root path of grand source, ex: /home/user/grand/grand

    @return (string) : root path of grand source
    """
    return osp.join(grand_get_path_root_pkg(), "grand")

# RK: delete this. Use 'path_grandlib_data'.
GRAND_DATA_PATH = osp.join(grand_get_path_root_pkg(), "data")

# RK: delete this. Use 'path_grandlib_data'.
def grand_add_path_data(s_file):
    return os.path.join(GRAND_DATA_PATH, s_file)

def path_grandlib_root():
    """get root path of grand source, ex: /home/user/grand

    @return (PosixPath object) : root path of grand source
    """
    return Path(__file__).parent.parent

def path_grandlib():
    """get root path of grand source, ex: /home/user/grand/grand

    @return (PosixPath object) : root path of grand source
    """
    return Path(__file__).parent

def path_grandlib_data(name=""):
    """get path of the grand folder, ex: /home/user/grand/data

    @return (PosixPath object) : path of the grand data folder.
    """
    if type(name)!=str:
        raise TypeError("Provide string as an input.")

    if name=="":
        return path_grandlib_root()/"data"
    elif name.lower()=="topography":
        # returns path to the topography folder
        return path_grandlib_root()/"data"/"topography"
    elif name.lower()=="geomagnet":
        # returns path to the geomagnet folder
        return path_grandlib_root()/"data"/"geomagnet"
    else:
        raise Exception(f"Provide a proper name for the requested data instead of '{name}'. Accepted names are topography and geomagnet.")

def grandcs_origin():
    # place-holder for the origin of the GRAND coordinate system to be used throughout the grandlib package.
    # Copied from https://www2-internet.iap.fr/grand/wikigrand/index.php?title=File:GRAND_GP65_RTK_positions.xls
    # NCS	0	40.95540292	93.97294689	1231.076	0	0	0	523.47	266.258	-4.275	523.57456	266.858	-2.206784
    latitude  = 40.95540292
    longitude = 93.97294689
    height    = 1231.076
    reference = 'Ellipsoid'

    return latitude, longitude, height, reference

#from grand.geo import geomagnet, topography
#from grand.geo.topography import geoid_undulation, Reference, Topography
#from grand.geo import coordinates
#from grand.geo.geomagnet import Geomagnet
#from grand.geo.coordinates import (
#    Coordinates,
#    CartesianRepresentation,
#    SphericalRepresentation,
#    GeodeticRepresentation,
#    Geodetic,
#    GRANDCS,
#    LTP,
#    ECEF,
#    HorizontalVector,
#    Horizontal,
#    HorizontalRepresentation,
#    Rotation,
#)
#import grand.dataio.protocol as store
#from grand.sim import efield2voltage
#from grand.sim.efield2voltage import Efield2Voltage
#from grand.sim.detector.antenna_model import tabulated_antenna_model, AntennaModel
#from grand.sim.detector.process_ant import AntennaProcessing
#from grand.sim.detector.rf_chain import RFChain
#from grand.sim.detector.adc import ADC
#from grand.sim.noise.galaxy import galactic_noise
#from grand.sim.shower.gen_shower import ShowerEvent
#from grand.sim.shower.pdg import ParticleCode


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




