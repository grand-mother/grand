from __future__ import annotations

from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Union, cast, Any
from numbers import Number
import os.path as osp

import numpy

from grand.io import io_node as io

__all__ = ["DataTable", "TabulatedAntennaModel"]


logger = getLogger(__name__)


@dataclass
class DataTable:
    frequency: Union[Number, numpy.ndarray]
    theta: Union[Number, numpy.ndarray]
    phi: Union[Number, numpy.ndarray]
    resistance: Union[Number, numpy.ndarray]
    reactance: Union[Number, numpy.ndarray]
    leff_theta: Union[Number, numpy.ndarray]
    phase_theta: Union[Number, numpy.ndarray]
    leff_phi: Union[Number, numpy.ndarray]
    phase_phi: Union[Number, numpy.ndarray]
    leff_phi_cart: Any = None
    leff_theta_cart: Any = None

    def __post_init__(self):
        self.phase_theta_rad = numpy.deg2rad(self.phase_theta)
        self.phase_phi_rad = numpy.deg2rad(self.phase_phi)

    def dump(self, node: io.DataNode) -> None:
        for field in fields(self):
            node.write(field.name, getattr(self, field.name), dtype="f4")

    @classmethod
    def load(cls, node: io.DataNode) -> DataTable:
        data = {}
        for field in fields(cls):
            data[field.name] = node.read(field.name)
        return DataTable(**data)

    def compute_leff_cartesian(self):
        """
        Swith Leff modulus, argument representation to cartesian (real and imaginary component)
        :param self:
        """
        leff_phi_cart = self.leff_phi * numpy.exp(1j * self.phase_phi_rad)
        self.leff_phi_cart = numpy.moveaxis(leff_phi_cart, 0, -1)
        delattr(self, "leff_phi")
        delattr(self, "phase_phi")
        delattr(self, "phase_phi_rad")
        leff_theta_cart = self.leff_theta * numpy.exp(1j * self.phase_theta_rad)
        self.leff_theta_cart = numpy.moveaxis(leff_theta_cart, 0, -1)
        delattr(self, "leff_theta")
        delattr(self, "phase_theta")
        delattr(self, "phase_theta_rad")
        logger.debug(f"self.leff_phi_cart: {self.leff_phi_cart.shape}")


@dataclass
class TabulatedAntennaModel(object):
    table: DataTable
    n_file: ... = "TBD"

    def __str__(self):
        ret = f"TabulatedAntennaModel, shape freq {self.table.frequency.shape}"
        ret += f"\nleff_theta: {self.table.leff_theta.shape} {self.table.leff_theta.dtype}"
        return ret

    def dump(self, destination: Union[str, Path, io.DataNode]) -> None:
        if type(destination) == io.DataNode:
            node = cast(io.DataNode, destination)
            self.table.dump(node)
        else:
            path = cast(Union[Path, str], destination)
            with io.open(path, "w") as node:
                self.table.dump(node)

    @classmethod
    def load(cls, source: Union[str, Path, io.DataNode]) -> TabulatedAntennaModel:
        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            filename = f"{source.filename}:{source.path}"
            loader = "_load_from_node"
        else:
            source = cast(Union[Path, str], source)
            filename = f"{source}:/"
            source = Path(source)
            if source.suffix == ".npy":
                loader = "_load_from_numpy"
            elif source.suffix == ".npz":
                loader = "_load_from_numpy_savez"
            else:
                loader = "_load_from_datafile"
        logger.info(f"Loading tabulated antenna model from {filename}")
        load = getattr(cls, loader)
        self = load(source)
        self.n_file = osp.basename(source)
        t = self.table
        n = t.frequency.size * t.theta.size * t.phi.size
        logger.info(f"Loaded {n} entries from {filename}")
        return self

    @classmethod
    def _load_from_datafile(cls, path: Union[Path, str]) -> TabulatedAntennaModel:
        with io.open(path) as root:
            return cls._load_from_node(root)

    @classmethod
    def _load_from_node(cls, node: io.DataNode) -> TabulatedAntennaModel:
        return cls(table=DataTable.load(node))

    @classmethod
    def _load_from_numpy(cls, path: Union[Path, str]) -> TabulatedAntennaModel:
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = numpy.load(path)
        n_f = f.shape[0]
        n_theta = len(numpy.unique(theta[0, :]))
        n_phi = int(R.shape[1] / n_theta)
        shape = (n_f, n_phi, n_theta)
        logger.debug(f"shape freq, phi, theta: {f.shape} {phi.shape} {theta.shape}")
        logger.debug(f"shape R, X: {R.shape} {X.shape} {R.dtype} {X.dtype}")
        logger.debug(f"shape module tetha : {lefft.shape}")
        logger.debug(f"shape arg tetha : {phaset.shape}")
        logger.debug(f"type leff  : {lefft.dtype}")
        logger.debug(f"type f  : {f.dtype}")
        logger.debug(f"type phi  : {phi.dtype}")
        logger.debug(f"min max resistance  : {R.min()} {R.max()}")
        logger.debug(f"min max reactance  : {X.min()} {X.max()}")
        dtype = "f4"
        f = f[:, 0].astype(dtype) * 1.0e6  # MHz --> Hz
        theta = theta[0, :n_theta].astype(dtype)  # deg
        phi = phi[0, ::n_theta].astype(dtype)  # deg
        R = R.reshape(shape).astype(dtype)  # Ohm
        X = X.reshape(shape).astype(dtype)  # Ohm
        lefft = lefft.reshape(shape).astype(dtype)  # m
        leffp = leffp.reshape(shape).astype(dtype)  # m
        # RK TODO: Make sure going from rad to deg does not affect calculations somewhere else.
        phaset = phaset.reshape(shape).astype(dtype)  # deg
        phasep = phasep.reshape(shape).astype(dtype)  # deg
        t = DataTable(
            frequency=f,
            theta=theta,
            phi=phi,
            resistance=R,
            reactance=X,
            leff_theta=lefft,
            phase_theta=phaset,
            leff_phi=leffp,
            phase_phi=phasep,
        )
        return cls(table=t)

    @classmethod
    def _load_from_numpy_savez(cls, path: Union[Path, str]) -> TabulatedAntennaModel:
        f_leff = numpy.load(path)
        if f_leff["version"][0] == "1.0":
            t_file = DataTable(
                frequency=f_leff["freq_mhz"] * 1e6,
                theta=numpy.arange(91).astype(float),
                phi=numpy.arange(361).astype(float),
                resistance=0,
                reactance=0,
                leff_theta=0,
                phase_theta=0,
                leff_phi=0,
                phase_phi=0,
                leff_phi_cart=f_leff["leff_phi"],
                leff_theta_cart=f_leff["leff_theta"],
            )
        else:
            raise
        return cls(table=t_file)
