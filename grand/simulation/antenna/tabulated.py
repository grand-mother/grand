from __future__ import annotations

from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Union, cast

from astropy.coordinates import BaseRepresentation, CartesianRepresentation,   \
                                PhysicsSphericalRepresentation
import astropy.units as u
import numpy
import h5py

from .generic import AntennaModel
from ... import io

import os
grand_astropy = True
try:
    if os.environ['GRAND_ASTROPY']=="0":
        grand_astropy=False
except:
    pass


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
            elif source.suffix == '.hdf5':
                loader = '_load_from_hdf5'
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

    @classmethod
    def _load_from_hdf5(cls, path: Union[Path, str], arm: str = "EW") -> TabulatedAntennaModel:
        print("tuutu")
        # Open the HDF5 file with antenna info
        ant_file = h5py.File(path, 'r')
        # Select the requested arm from the file
        ant_arm = ant_file[arm]
        # Load the antenna parameters (to numpy arrays, thus [:])
        f = ant_arm["frequency"][:]
        R = ant_arm["resistance"][:]
        X = ant_arm["reactance"][:]
        theta = ant_arm["theta"][:]
        phi = ant_arm["phi"][:]
        lefft = ant_arm["leff_theta"][:]
        leffp = ant_arm["leff_phi"][:]
        phaset = ant_arm["phase_theta"][:]
        phasep = ant_arm["phase_phi"][:]
        print("all shape", f.shape, R.shape, X.shape, theta.shape, phi.shape, lefft.shape, leffp.shape, phaset.shape, phasep.shape)
        
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
        
        # LWP: Replace with manual conversion to spherical, assuming direction is a numpy array
        if grand_astropy:
            direction = direction.represent_as(PhysicsSphericalRepresentation)
            theta, phi = direction.theta, direction.phi
        else:
            theta = numpy.arctan2(numpy.sqrt(direction[0]**2+direction[1]**2),direction[2])
            phi = numpy.arctan2(direction[1],direction[0])

        # Interpolate using a tri-linear interpolation in (f, phi, theta)
        t = self.table

        dtheta = t.theta[1] - t.theta[0]
        # LWP: subtracting values from numpy array
        if grand_astropy:
            rt1 = ((theta - t.theta[0]) / dtheta).to_value(u.one)
        else:
            rt1 = ((theta - t.theta[0].to_value('rad')) / dtheta.to_value('rad'))
        it0 = int(numpy.floor(rt1) % t.theta.size)
        it1 = it0 + 1
        if it1 == t.theta.size: # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= numpy.floor(rt1)
        rt0 = 1 - rt1

        dphi = t.phi[1] - t.phi[0]
        # LWP: subtracting values from numpy array
        if grand_astropy:
            rp1 = ((phi - t.phi[0]) / dphi).to_value(u.one)
        else:
            rp1 = ((phi - t.phi[0].to_value('rad')) / dphi.to_value('rad'))
        
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

	# LWP: to_value not needed anymore
        if grand_astropy:
            t, p = theta.to_value('rad'), phi.to_value('rad')
        else:
            t, p = theta, phi
        
        ct, st = numpy.cos(t), numpy.sin(t)
        cp, sp = numpy.cos(p), numpy.sin(p)
        lx = lt * ct * cp - sp * lp
        ly = lt * ct * sp + cp * lp
        lz = -st * lt

        return CartesianRepresentation(lx, ly, lz, unit='m')
