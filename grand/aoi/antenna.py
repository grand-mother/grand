# Created by Lech Wiktor Piotrowski at 14/03/2025
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grand import CartesianRepresentation


@dataclass
class Antenna:
    """A class describing a single antenna"""

    id: int = -1
    """Antenna ID - the du_id from the trees"""

    ## Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)
    # position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    _position: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Antenna tilt
    _tilt: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))
    ## Antenna acceleration - this comes from hardware. ToDo: perhaps recalculate to tilt or remove tilt?
    _acceleration: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(1, np.float64), y=np.zeros(1, np.float64), z=np.zeros(1, np.float64)))

    model: Any = 0
    """The antenna model"""

    # # ToDo: Parameters below come from the hardware, but do we want them here?
    # ## Atmospheric temperature (read via I2C)
    # atm_temperature: float = 0
    # ## Atmospheric pressure
    # atm_pressure: float = 0
    # ## Atmospheric humidity
    # atm_humidity: float = 0
    # ## Battery voltage
    # battery_level: float = 0
    # ## Firmware version
    # firmware_version: float = 0

    @property
    def position(self):
        """Antenna position in site's referential (x = SN, y=EW,  0 = center of array + sea level)"""
        return self._position

    @position.setter
    def position(self, v):
        r"""Sets the antenna position.

                Parameters
                ----------
                v : array_like
                    Position in the array frame, in metres.
        """
        self._position = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def tilt(self):
        """Antenna tilt"""
        return self._tilt

    @tilt.setter
    def tilt(self, v):
        r"""Sets the antenna tilt.

                Parameters
                ----------
                v : array_like
                    Tilt angles, in degrees.
        """
        self._tilt = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def acceleration(self):
        """Antenna acceleration - this comes from hardware."""
        return self._acceleration

    @acceleration.setter
    def acceleration(self, v):
        r"""Sets the measured acceleration.

                Parameters
                ----------
                v : array_like
                    Acceleration components, used to infer the tilt.
        """
        self._acceleration = CartesianRepresentation(x=v[0], y=v[1], z=v[2])
