from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Union

from astropy.coordinates import BaseRepresentation, CartesianRepresentation
import astropy.units as u
import numpy

from ... import io, ECEF, LTP

__all__ = ["Antenna", "AntennaModel", "ElectricField", "Voltage"]


_logger = getLogger(__name__)


@dataclass
class ElectricField:
    t: u.Quantity
    E: Union[ECEF, LTP]

    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f"Loading E-field from {node.filename}:{node.path}")
        t = node.read("t", dtype="f8")
        E = node.read("E", dtype="f8")
        # XXX read the frame
        return cls(t, E)

    def dump(self, node: io.DataNode):
        _logger.debug(f"Dumping E-field to {node.filename}:{node.path}")
        # XXX store the frame
        node.write("t", self.t, unit="ns", dtype="f4")
        node.write("E", self.E.cartesian, unit="uV/m", dtype="f4")


@dataclass
class Voltage:
    t: u.Quantity
    V: u.Quantity

    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f"Loading voltage from {node.filename}:{node.path}")
        t = node.read("t", dtype="f8")
        V = node.read("V", dtype="f8")
        return cls(t, V)

    def dump(self, node: io.DataNode):
        _logger.debug(f"Dumping E-field to {node.filename}:{node.path}")
        node.write("t", self.t, unit="ns", dtype="f4")
        node.write("V", self.V, unit="uV", dtype="f4")


class AntennaModel:
    def effective_length(self, direction: BaseRepresentation,
        frequency: u.Quantity) -> CartesianRepresentation:
        pass


@dataclass
class Antenna:
    model: AntennaModel
    frame: Union[ECEF, LTP, None] = None

    def compute_voltage(self, direction: Union[ECEF, LTP],
        field: ElectricField) -> Voltage:

        def rfft(q):
            return numpy.fft.rfft(q.value) * q.unit

        def irfft(q):
            return numpy.fft.irfft(q.value) * q.unit

        def fftfreq(n, t):
            dt = (t[1] - t[0]).to_value("s")
            return numpy.fft.fftfreq(n, dt) * u.Hz

        Ex = rfft(field.E.x)
        Ey = rfft(field.E.y)
        Ez = rfft(field.E.z)
        f = fftfreq(Ex.size, field.t)

        if self.frame:
            direction = direction.transform_to(self.frame)

        Leff:CartesianRepresentation
        Leff = self.model.effective_length(direction.cartesian, f)

        if self.frame:
            tmp = self.frame(Leff, copy=False)
            tmp = tmp.transform_to(field.E)
            Leff = tmp.cartesian

        V = irfft(Ex * Leff.x + Ey * Leff.y + Ez * Leff.z)
        t = field.t
        t = t[:V.size]

        return Voltage(t=t, V=V)
