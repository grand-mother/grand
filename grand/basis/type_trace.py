from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Union

import numpy as np

from grand.geo.coordinates import (
    LTP,
    GRANDCS,
    CartesianRepresentation,
)
from grand.io import io_node as io  # , ECEF, LTP


logger = getLogger(__name__)


@dataclass
class ElectricField:
    a_time: np.ndarray
    e_xyz: CartesianRepresentation  # RK
    pos_xyz: Union[CartesianRepresentation, None] = None
    frame: Union[LTP, GRANDCS, None] = None

    def __post_init__(self):
        assert self.a_time.shape[0] == self.e_xyz.shape[1]

    @classmethod
    def load(cls, node: io.DataNode):
        logger.debug(f"Loading E-field from {node.filename}:{node.path}")

        a_time = node.read("t", dtype="f8")
        e_xyz = node.read("E", dtype="f8")

        try:
            par_r = node.read("r", dtype="f8")
        except KeyError:
            par_r = None

        try:
            frame = node.read("frame")
        except KeyError:
            frame = None

        return cls(a_time, e_xyz, par_r, frame)

    def dump(self, node: io.DataNode):
        logger.debug(f"Dumping E-field to {node.filename}:{node.path}")

        node.write("t", self.a_time, dtype="f4")
        node.write("E", self.e_xyz, dtype="f4")

        if self.pos_xyz is not None:
            node.write("r", self.pos_xyz, dtype="f4")

        if self.frame is not None:
            node.write("frame", self.frame)


@dataclass
class Voltage:
    t: np.ndarray  # [s]
    V: np.ndarray  # [?]

    @classmethod
    def load(cls, node: io.DataNode):
        logger.debug("Loading voltage from {node.filename}:{node.path}")
        t = node.read("t", dtype="f8")
        V = node.read("V", dtype="f8")
        return cls(t, V)

    def dump(self, node: io.DataNode):
        logger.debug("Dumping E-field to {node.filename}:{node.path}")
        node.write("t", self.t, dtype="f4")
        node.write("V", self.V, dtype="f4")
