from __future__ import annotations

from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Union, cast

from astropy.coordinates import BaseRepresentation, CartesianRepresentation,   \
                                PhysicsSphericalRepresentation
import astropy.units as u
import numpy

from .generic import AntennaModel
from ... import io

__all__ = ['DataTable', 'TabulatedAntennaModel']


_logger = getLogger(__name__)


@dataclass
class DataTable:
    frequency: u.Quantity
    theta: u.Quantity
    phi: u.Quantity
    resistance: u.Quantity
    reactance: u.Quantity
    leff_theta: u.Quantity
    phase_theta: numpy.ndarray
    leff_phi: u.Quantity
    phase_phi: numpy.ndarray

    def dump(self, node: io.DataNode) -> None:
        for field in fields(self):
            node.write(field.name, getattr(self, field.name), dtype='f4')

    @classmethod
    def load(cls, node: io.DataNode) -> DataTable:
        data = {}
        for field in fields(cls):
            data[field.name] = node.read(field.name)
        return DataTable(**data)


@dataclass
class TabulatedAntennaModel(AntennaModel):
    table: DataTable

    def dump(self, destination: Union[str, Path, io.DataNode]) -> None:
        if type(destination) == io.DataNode:
            node = cast(io.DataNode, destination)
            self.table.dump(node)
        else:
            path = cast(Union[Path, str], destination)
            with io.open(path, 'w') as node:
                self.table.dump(node)

    @classmethod
    def load(cls, source: Union[str, Path, io.DataNode])                       \
        -> TabulatedAntennaModel:

        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            filename = f'{source.filename}:{source.path}'
            loader = '_load_from_node'
        else:
            source = cast(Union[Path, str], source)
            filename = f'{source}:/'
            source = Path(source)
            if source.suffix == '.npy':
                loader = '_load_from_numpy'
            else:
                loader = '_load_from_datafile'

        _logger.info(f'Loading tabulated antenna model from {filename}')

        load = getattr(cls, loader)
        self = load(source)

        t = self.table
        n = t.frequency.size * t.theta.size * t.phi.size
        _logger.info(f'Loaded {n} entries from {filename}')

        return self

    @classmethod
    def _load_from_datafile(cls, path: Union[Path, str])                       \
        -> TabulatedAntennaModel:

        with io.open(path) as root:
            return cls._load_from_node(root)

    @classmethod
    def _load_from_node(cls, node: io.DataNode) -> TabulatedAntennaModel:
        return cls(table = DataTable.load(node))

    @classmethod
    def _load_from_numpy(cls, path: Union[Path, str]) -> TabulatedAntennaModel:
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = numpy.load(path)

        n_f = f.shape[0]
        n_theta = len(numpy.unique(theta[0,:]))
        n_phi = int(R.shape[1] / n_theta)
        shape = (n_f, n_phi, n_theta)

        dtype = 'f4'
        f = f[:,0].astype(dtype) * u.MHz
        theta = theta[0, :n_theta].astype(dtype) * u.deg
        phi = phi[0, ::n_theta].astype(dtype) * u.deg
        R = R.reshape(shape).astype(dtype) * u.Ohm
        X = X.reshape(shape).astype(dtype) * u.Ohm
        lefft = lefft.reshape(shape).astype(dtype) * u.m
        leffp = leffp.reshape(shape).astype(dtype) * u.m
        phaset = numpy.deg2rad(phaset).reshape(shape).astype(dtype) * u.rad
        phasep = numpy.deg2rad(phasep).reshape(shape).astype(dtype) * u.rad

        t = DataTable(frequency = f, theta = theta, phi = phi, resistance = R,
                      reactance = X, leff_theta = lefft, phase_theta = phaset,
                      leff_phi = leffp, phase_phi = phasep)
        return cls(table=t)

    def effective_length(self, direction: BaseRepresentation,
        frequency: u.Quantity) -> CartesianRepresentation:

        direction = direction.represent_as(PhysicsSphericalRepresentation)
        theta, phi = direction.theta, direction.phi

        # Interpolate using a tri-linear interpolation in (f, phi, theta)
        t = self.table

        dtheta = t.theta[1] - t.theta[0]
        rt1 = ((theta - t.theta[0]) / dtheta).to_value(u.one)
        it0 = int(numpy.floor(rt1) % t.theta.size)
        it1 = it0 + 1
        if it1 == t.theta.size: # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= numpy.floor(rt1)
        rt0 = 1 - rt1

        dphi = t.phi[1] - t.phi[0]
        rp1 = ((phi - t.phi[0]) / dphi).to_value(u.one)
        ip0 = int(numpy.floor(rp1) % t.phi.size)
        ip1 = ip0 + 1
        if ip1 == t.phi.size: # Results are periodic along phi
            ip1 = 0
        rp1 -= numpy.floor(rp1)
        rp0 = 1 - rp1

        x = frequency.to_value('Hz')
        xp = t.frequency.to_value('Hz')

        def interp(v):
            fp = rp0 * rt0 * v[:, ip0, it0] + rp1 * rt0 * v[:, ip1, it0] +     \
                 rp0 * rt1 * v[:, ip0, it1] + rp1 * rt1 * v[:, ip1, it1]
            return numpy.interp(x, xp, fp, left=0, right=0)

        ltr = interp(t.leff_theta.to_value('m'))
        lta = interp(t.phase_theta.to_value('rad'))
        lpr = interp(t.leff_phi.to_value('m'))
        lpa = interp(t.phase_phi.to_value('rad'))

        # Pack the result as a Cartesian vector with complex values
        lt = ltr * numpy.exp(1j * lta)
        lp = lpr * numpy.exp(1j * lpa)

        t, p = theta.to_value('rad'), phi.to_value('rad')
        ct, st = numpy.cos(t), numpy.sin(t)
        cp, sp = numpy.cos(p), numpy.sin(p)
        lx = lt * ct * cp - sp * lp
        ly = lt * ct * sp + cp * lp
        lz = -st * lt

        return CartesianRepresentation(lx, ly, lz, unit='m')
