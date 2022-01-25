from __future__ import annotations

from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Union, cast
from numbers import Number

import numpy

from .generic import AntennaModel
from grand.io import hdf5

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

    def dump(self, node: hdf5.DataNode) -> None:
        for field in fields(self):
            node.write(field.name, getattr(self, field.name), dtype="f4")

    @classmethod
    def load(cls, node: hdf5.DataNode) -> DataTable:
        data = {}
        for field in fields(cls):
            data[field.name] = node.read(field.name)
        return DataTable(**data)


@dataclass
class TabulatedAntennaModel(AntennaModel):
    table: DataTable

    def dump(self, destination: Union[str, Path, hdf5.DataNode]) -> None:
        if type(destination) == hdf5.DataNode:
            node = cast(hdf5.DataNode, destination)
            self.table.dump(node)
        else:
            path = cast(Union[Path, str], destination)
            with hdf5.open(path, "w") as node:
                self.table.dump(node)

    @classmethod
    def load(cls, source: Union[str, Path, hdf5.DataNode]) -> TabulatedAntennaModel:

        if type(source) == hdf5.DataNode:
            source = cast(hdf5.DataNode, source)
            filename = f"{source.filename}:{source.path}"
            loader = "_load_from_node"
        else:
            source = cast(Union[Path, str], source)
            filename = f"{source}:/"
            source = Path(source)
            if source.suffix == ".npy":
                loader = "_load_from_numpy"
            else:
                loader = "_load_from_datafile"

        logger.info(f"Loading tabulated antenna model from {filename}")

        load = getattr(cls, loader)
        self = load(source)

        t = self.table
        n = t.frequency.size * t.theta.size * t.phi.size
        logger.info(f"Loaded {n} entries from {filename}")

        return self

    @classmethod
    def _load_from_datafile(cls, path: Union[Path, str]) -> TabulatedAntennaModel:

        with hdf5.open(path) as root:
            return cls._load_from_node(root)

    @classmethod
    def _load_from_node(cls, node: hdf5.DataNode) -> TabulatedAntennaModel:
        return cls(table=DataTable.load(node))

    @classmethod
    def _load_from_numpy(cls, path: Union[Path, str]) -> TabulatedAntennaModel:
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = numpy.load(path)

        n_f = f.shape[0]
        n_theta = len(numpy.unique(theta[0, :]))
        n_phi = int(R.shape[1] / n_theta)
        shape = (n_f, n_phi, n_theta)

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
