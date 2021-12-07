"""Tools for the GRAND package
"""

from typing_extensions import Final

import os
from pathlib import Path

# from .geomagnet import Geomagnet
# from .coordinates import *

# RK
# from .coordinates import CartesianRepresentation, SphericalRepresentation, ECEF, Geodetic, GeodeticRepresentation, GRAND, LTP
# from .geomagnet import *
# from .topography import geoid_undulation, Reference, Topography

__all__ = ["DATADIR"]
# RK
# __all__ = ['DATADIR', 'CartesianRepresentation', 'ECEF',
# 		   'Geodetic', 'GeodeticRepresentation', 'GRANDCS',
# 		   'LTP']

# Initialise the package globals
DATADIR: Final = Path(__file__).parent / "data"
"""Path to the package data"""
