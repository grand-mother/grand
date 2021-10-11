from __future__ import annotations

from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Union, cast
from numbers import Number

import numpy

from .generic import AntennaModel
from ... import io

from ...tools.coordinates import CartesianRepresentation, SphericalRepresentation 

__all__ = ['DataTable', 'TabulatedAntennaModel']


_logger = getLogger(__name__)


@dataclass
class DataTable:
    frequency   : Number
    theta       : Number
    phi         : Number
    resistance  : Number
    reactance   : Number
    leff_theta  : Number
    phase_theta : numpy.ndarray
    leff_phi    : Number
    phase_phi   : numpy.ndarray

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

        n_f     = f.shape[0]
        n_theta = len(numpy.unique(theta[0,:]))
        n_phi   = int(R.shape[1] / n_theta)
        shape   = (n_f, n_phi, n_theta)

        dtype   = 'f4'
        f       = f[:,0].astype(dtype)*1.e6          # MHz --> Hz
        theta   = theta[0, :n_theta].astype(dtype)   # deg
        phi     = phi[0, ::n_theta].astype(dtype)    # deg
        R       = R.reshape(shape).astype(dtype)     # Ohm
        X       = X.reshape(shape).astype(dtype)     # Ohm
        lefft   = lefft.reshape(shape).astype(dtype) # m
        leffp   = leffp.reshape(shape).astype(dtype) # m

        # RK TODO: Make sure going from rad to deg does not affect calculations somewhere else.
        phaset  = phaset.reshape(shape).astype(dtype) # deg 
        phasep  = phasep.reshape(shape).astype(dtype) # deg

        t = DataTable(frequency = f, theta = theta, phi = phi, resistance = R,
                      reactance = X, leff_theta = lefft, phase_theta = phaset,
                      leff_phi = leffp, phase_phi = phasep)
        return cls(table=t)

    def effective_lengthX(self, direction: CartesianRepresentation,
        frequency: Number) -> CartesianRepresentation:

        direction_cart = CartesianRepresentation(direction) #RK
        direction_sphr = SphericalRepresentation(direction_cart)
        theta, phi     = direction_sphr.theta, direction_sphr.phi

        print('Direction to Xmax in antenna frame [m]  :', direction.flatten())
        print('Direction to Xmax in antenna frame [deg]:', theta, 360.+phi)
        print('---------------------------------', '\n')
        # Interpolate using a tri-linear interpolation in (f, phi, theta)
        t = self.table

        #print('table:', t)
        
        #dtheta = t.theta.value[1] - t.theta.value[0]
        dtheta = t.theta[1] - t.theta[0] #RK. t.theta is in deg. remove .value after astropy is completely removed.
        #rt1 = ((theta - t.theta[0]) / dtheta).to_value(u.one)
        rt1 = ((theta - t.theta[0]) / dtheta) # LWP: subtracting values from numpy array
        it0 = int(numpy.floor(rt1) % t.theta.size)
        it1 = it0 + 1
        if it1 == t.theta.size: # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= numpy.floor(rt1)
        rt0 = 1 - rt1

        dphi = t.phi[1] - t.phi[0]      # RK. t.phi is in deg.
        #rp1 = ((phi - t.phi[0]) / dphi).to_value(u.one)
        rp1 = ((phi - t.phi[0]) / dphi)  # LWP: subtracting values from numpy array
        ip0 = int(numpy.floor(rp1) % t.phi.size)
        ip1 = ip0 + 1
        if ip1 == t.phi.size: # Results are periodic along phi
            ip1 = 0
        rp1 -= numpy.floor(rp1)
        rp0 = 1 - rp1

        #x = frequency.to_value('Hz')
        #xp = t.frequency.to_value('Hz')
        x  = frequency    # LWP. Hz
        xp = t.frequency  # LWP. Hz 
        def interp(v):
            fp = rp0 * rt0 * v[:, ip0, it0] + rp1 * rt0 * v[:, ip1, it0] +     \
                 rp0 * rt1 * v[:, ip0, it1] + rp1 * rt1 * v[:, ip1, it1]
            return numpy.interp(x, xp, fp, left=0, right=0)



        #ltr = interp(t.leff_theta.to_value('m'))
        #lta = interp(t.phase_theta.to_value('rad'))
        #lpr = interp(t.leff_phi.to_value('m'))
        #lpa = interp(t.phase_phi.to_value('rad'))
        ltr = interp(t.leff_theta)                 # LWP. m 
        lta = interp(numpy.deg2rad(t.phase_theta)) # LWP. rad 
        lpr = interp(t.leff_phi)                   # LWP. m 
        lpa = interp(numpy.deg2rad(t.phase_phi))   # LWP. rad 
        
        # Pack the result as a Cartesian vector with complex values
        lt = ltr * numpy.exp(1j * lta)
        lp = lpr * numpy.exp(1j * lpa)

        '''
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(211)
        labs = print
        plt.plot(t.frequency/1e6, t.leff_theta[:,ip0,it0],'--',label=f'Tabulated at theta={t.theta[it0]}')
        plt.plot(t.frequency/1e6, t.leff_theta[:,ip1,it1],'--',label=f'Tabulated at theta={t.theta[it1]}')
        plt.plot(x[x>0]/1e6,numpy.abs(lt[x>0]),label=f'Interpolated at theta={theta}')
        #plt.xlabel("Frequency (MHz)")
        plt.xlabel("")
        plt.ylabel("|Leff theta| (m)")
        plt.legend(loc='best')
        plt.grid(ls='--', alpha=0.3)
        plt.subplot(212)
        plt.plot(t.frequency/1e6, t.leff_phi[:,ip0,it0],'--',label=f'Tabulated at phi={t.phi[ip0]}')
        plt.plot(t.frequency/1e6, t.leff_phi[:,ip1,it1],'--',label=f'Tabulated at phi={t.phi[ip1]}')
        plt.plot(x[x>0]/1e6,numpy.abs(lp[x>0]),label=f'Interpolated at phi={phi}')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("|Leff phi| (m)")
        plt.legend(loc='best')
        plt.grid(ls='--', alpha=0.3)
        plt.show()
        numpy.savetxt('lefft_new.txt',numpy.abs(lt[x>0]))
        numpy.savetxt('leffp_new.txt',numpy.abs(lp[x>0]))
        numpy.savetxt('f_new.txt',x[x>0])'''

        #t, p = theta.to_value('rad'), phi.to_value('rad')
        t, p   = numpy.deg2rad(theta), numpy.deg2rad(phi)
        ct, st = numpy.cos(t), numpy.sin(t)
        cp, sp = numpy.cos(p), numpy.sin(p)
        lx     = lt * ct * cp - sp * lp
        ly     = lt * ct * sp + cp * lp
        lz     = -st * lt
        
        
        '''plt.figure()
        plt.plot(x[x>0]/1e6,numpy.abs(lx)[x>0],label='Xant=SN')
        plt.plot(x[x>0]/1e6,numpy.abs(ly)[x>0],label='Yant=EW')
        plt.plot(x[x>0]/1e6,numpy.abs(lz)[x>0],label='Zant=Up')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("|Leff| (m)")
        plt.legend(loc="best")
        plt.show()'''

        #return CartesianRepresentation(lx, ly, lz, unit='m')
        return CartesianRepresentation(x=lx, y=ly, z=lz)

