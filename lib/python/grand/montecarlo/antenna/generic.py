from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Optional

from astropy.coordinates import BaseRepresentation, CartesianRepresentation
import astropy.units as u
import numpy

from ... import io

__all__ = ["Antenna", "AntennaModel", "ElectricField", "Voltage"]


_logger = getLogger(__name__)


@dataclass
class ElectricField:
    t: u.Quantity
    E: CartesianRepresentation

    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f"Loading E-field from {node.filename}:{node.path}")
        t = node.read("t", dtype="f8")
        E = node.read("E", dtype="f8")
        return cls(t, E)

    def dump(self, node: io.DataNode):
        _logger.debug(f"Dumping E-field to {node.filename}:{node.path}")
        node.write("t", self.t, unit="ns", dtype="f4")
        node.write("E", self.E, unit="uV/m", dtype="f4")


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

    def compute_voltage(self, direction: BaseRepresentation,
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

        Leff = self.model.effective_length(direction, f)

        V = irfft(Ex * Leff.x + Ey * Leff.y + Ez * Leff.z)
        t = field.t
        t = t[:V.size]

        return Voltage(t=t, V=V)
