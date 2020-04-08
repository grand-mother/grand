from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import cast, Optional, Union

from astropy.coordinates import BaseRepresentation, CartesianRepresentation
import astropy.units as u
import numpy

from ... import io, ECEF, LTP

__all__ = ['Antenna', 'AntennaModel', 'ElectricField', 'MissingFrameError',
           'Voltage']


_logger = getLogger(__name__)


@dataclass
class ElectricField:
    t: u.Quantity
    E: BaseRepresentation
    r: Union[BaseRepresentation, None] = None
    frame: Union[ECEF, LTP, None] = None

    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f'Loading E-field from {node.filename}:{node.path}')

        t = node.read('t', dtype='f8')
        E = node.read('E', dtype='f8')

        try:
            r = node.read('r', dtype='f8')
        except KeyError:
            r = None

        try:
            frame = node.read('frame')
        except KeyError:
            frame = None

        return cls(t, E, r, frame)

    def dump(self, node: io.DataNode):
        _logger.debug(f'Dumping E-field to {node.filename}:{node.path}')

        node.write('t', self.t, unit='ns', dtype='f4')
        node.write('E', self.E, unit='uV/m', dtype='f4')

        if self.r is not None:
            node.write('r', self.r, unit='m', dtype='f4')

        if self.frame is not None:
            node.write('frame', self.frame)


@dataclass
class Voltage:
    t: u.Quantity
    V: u.Quantity

    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f'Loading voltage from {node.filename}:{node.path}')
        t = node.read('t', dtype='f8')
        V = node.read('V', dtype='f8')
        return cls(t, V)

    def dump(self, node: io.DataNode):
        _logger.debug(f'Dumping E-field to {node.filename}:{node.path}')
        node.write('t', self.t, unit='ns', dtype='f4')
        node.write('V', self.V, unit='uV', dtype='f4')


class AntennaModel:
    def effective_length(self, direction: BaseRepresentation,
        frequency: u.Quantity) -> CartesianRepresentation:
        pass


class MissingFrameError(ValueError):
    pass


@dataclass
class Antenna:
    model: AntennaModel
    frame: Union[ECEF, LTP, None] = None

    def compute_voltage(self, direction: Union[ECEF, LTP, BaseRepresentation],
            field: ElectricField, frame: Union[ECEF, LTP, None]=None)          \
            -> Voltage:

        # Uniformise the inputs
        if self.frame is None:
            antenna_frame = None
            if (frame is not None) or                                          \
               (not isinstance(field.E, BaseRepresentation)) or                \
               (not isinstance(direction, BaseRepresentation)):
                raise MissingFrameError('missing antenna frame')
            else:
                E_frame, dir_frame = None, None
                E = field.E
        else:
            antenna_frame = cast(Union[ECEF, LTP], self.frame)
            frame_required = False
            if field.frame is None:
                E_frame, frame_required = frame, True
            else:
                E_frame = field.frame

            if isinstance(direction, BaseRepresentation):
                dir_frame, frame_required = frame, True
            else:
                dir_frame = direction

            if frame_required and (frame is None):
                raise MissingFrameError('missing frame')

        # Compute the voltage
        def rfft(q):
            return numpy.fft.rfft(q.value) * q.unit

        def irfft(q):
            return numpy.fft.irfft(q.value) * q.unit

        def fftfreq(n, t):
            dt = (t[1] - t[0]).to_value('s')
            return numpy.fft.fftfreq(n, dt) * u.Hz

        E = field.E.represent_as(CartesianRepresentation)
        Ex = rfft(E.x)
        Ey = rfft(E.y)
        Ez = rfft(E.z)
        f = fftfreq(Ex.size, field.t)

        if dir_frame is not None:
            # Change the direction to the antenna frame
            if isinstance(direction, BaseRepresentation):
                direction = dir_frame.realize_frame(direction)
            direction = direction.transform_to(antenna_frame).data

        Leff:CartesianRepresentation
        Leff = self.model.effective_length(direction, f)

        if antenna_frame is not None:
            # Change the effective length to the E-field frame
            tmp = antenna_frame.realize_frame(Leff)
            tmp = tmp.transform_to(E_frame)
            Leff = tmp.cartesian

        V = irfft(Ex * Leff.x + Ey * Leff.y + Ez * Leff.z)
        t = field.t
        t = t[:V.size]

        return Voltage(t=t, V=V)
