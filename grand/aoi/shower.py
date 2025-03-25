# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field

import numpy as np

from grand import CartesianRepresentation


@dataclass
class Shower:
    """A class for holding a shower"""

    energy_em: float = 0
    """Shower from e+- (ie related to radio emission) (GeV)"""

    energy_primary: float = 0
    """Total energy of the primary (including muons, neutrinos, ...) (GeV)"""

    Xmax: float = 0
    """Shower Xmax [g/cm2]"""

    _Xmaxpos: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    """Shower position in the site's reference frame"""

    azimuth: float = 0
    """Shower azimuth  (coordinates system = NWU + origin = core, "pointing to")"""

    zenith: float = 0
    """Shower zenith  (coordinates system = NWU + origin = core, , "pointing to")"""

    ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
    _origin_geoid: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Position of the core on the ground in the site's reference frame
    _core_ground_pos: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))

    @property
    def Xmaxpos(self):
        """Shower position in the site's reference frame"""
        return self._Xmaxpos

    @Xmaxpos.setter
    def Xmaxpos(self, v):
        self._Xmaxpos = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def origin_geoid(self):
        """Direction of origin"""
        return self._origin_geoid

    @origin_geoid.setter
    def origin_geoid(self, v):
        self._origin_geoid = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def core_ground_pos(self):
        """Position of the core on the ground in the site's reference frame"""
        return self._core_ground_pos

    @core_ground_pos.setter
    def core_ground_pos(self, v):
        self._core_ground_pos = CartesianRepresentation(x=v[0], y=v[1], z=v[2])
