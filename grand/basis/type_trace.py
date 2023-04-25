from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Union

import numpy as np
import scipy.fft as sf

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
        self.fft_e_3d = np.zeros((3, 0))
        assert self.a_time.shape[0] == self.e_xyz.shape[1]

    def get_fft(self, size_sig_pad):
        """

        :param size_sig_pad:
        :type size_sig_pad:
        """
        if self.fft_e_3d.size > 0:
            return self.fft_e_3d
        else:
            self.fft_e_3d = sf.rfft(self.e_xyz, n=size_sig_pad)
            return self.fft_e_3d

    def get_delta_time_s(self):
        return self.a_time[1] - self.a_time[0]

    def get_nb_sample(self):
        return self.e_xyz.shape[1]

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
