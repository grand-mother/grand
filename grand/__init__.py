'''GRAND software package
'''

from .tools import geomagnet, topography
from .tools.coordinates import ECEF, ExtendedCoordinateFrame, \
                               GeodeticRepresentation, \
                               HorizontalRepresentation, LTP, Rotation
from .logging import getLogger, Logger
from . import logging, store
import os.path as osp
from pathlib import Path

GRAND_DATA = osp.join(Path.home(), ".grand")

__all__ = ['geomagnet', 'getLogger', 'store', 'topography', 'ECEF',
           'ExtendedCoordinateFrame', 'GeodeticRepresentation',
           'HorizontalRepresentation', 'Logger', 'LTP', 'Rotation', "GRAND_DATA"]

logger:Logger = getLogger(__name__)
