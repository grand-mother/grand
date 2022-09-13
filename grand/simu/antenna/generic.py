from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Union, Any

import numpy as np

from grand.geo.coordinates import (
    ECEF,
    LTP,
    GRANDCS,
    CartesianRepresentation,
    SphericalRepresentation,
)  # RK


from grand.io import io_node as io  # , ECEF, LTP


logger = getLogger(__name__)


@dataclass
class ElectricField:
    t: np.ndarray
    E: CartesianRepresentation  # RK
    r: Union[CartesianRepresentation, None] = None
    frame: Union[LTP, GRANDCS, None] = None

    @classmethod
    def load(cls, node: io.DataNode):
        logger.debug(f"Loading E-field from {node.filename}:{node.path}")

        t = node.read("t", dtype="f8")
        E = node.read("E", dtype="f8")

        try:
            r = node.read("r", dtype="f8")
        except KeyError:
            r = None

        try:
            frame = node.read("frame")
        except KeyError:
            frame = None

        return cls(t, E, r, frame)

    def dump(self, node: io.DataNode):
        logger.debug(f"Dumping E-field to {node.filename}:{node.path}")

        node.write("t", self.t, dtype="f4")
        node.write("E", self.E, dtype="f4")

        if self.r is not None:
            node.write("r", self.r, dtype="f4")

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


class AntennaModel:
    # TODO: suspicious code no constructor , no test, dead code ?
    def effective_length(self) -> CartesianRepresentation:
        return CartesianRepresentation(0)


class MissingFrameError(ValueError):
    pass


@dataclass
class Antenna:
    model: Any  # if class is used, circular import error occurs.
    frame: Union[LTP, GRANDCS]

    def effective_length(
        self,
        xmax: LTP,
        Efield: ElectricField,
        frame: Union[LTP, GRANDCS],
    ) -> CartesianRepresentation:
        def fftfreq(n, t):
            dt = t[1] - t[0]
            return np.fft.fftfreq(n, dt)

        def interp(v):
            # fmt: off
            fp =  rp0*rt0*v[:, ip0, it0] \
                + rp1*rt0*v[:, ip1, it0] \
                + rp0*rt1*v[:, ip0, it1] \
                + rp1*rt1*v[:, ip1, it1]
            # fmt: on
            return np.interp(x, xp, fp, left=0, right=0)

        # 'frame' is shower frame. 'self.frame' is antenna frame.
        if isinstance(xmax, LTP):
            # shower frame --> antenna frame
            direction = xmax.ltp_to_ltp(self.frame)
        else:
            raise TypeError("Provide Xmax in LTP frame instead of %s" % type(xmax))
        # compute Efield direction in spherical coordinate in antenna frame
        direction_cart = CartesianRepresentation(direction)
        direction_sphr = SphericalRepresentation(direction_cart)
        theta_efield, phi_efield = direction_sphr.theta, direction_sphr.phi
        logger.debug(f"type theta_efield: {type(theta_efield)} {theta_efield}")
        # logger.debug(f"{theta_efield.r}")
        # Interpolate using a tri-linear interpolation in (f, phi, theta)
        # table store antenna response
        table = self.model.table
        # delta theta in degree
        dtheta = table.theta[1] - table.theta[0]
        # theta_efield between index it0 and it1 in theta antenna response representation
        rt1 = (theta_efield - table.theta[0]) / dtheta
        # prevent > 360 deg or >180 deg ?
        it0 = int(np.floor(rt1) % table.theta.size)
        it1 = it0 + 1
        if it1 == table.theta.size:  # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= np.floor(rt1)
        rt0 = 1 - rt1
        # phi_Efield between index ip0 and ip1 in phi antenna response representation
        dphi = table.phi[1] - table.phi[0]  # deg
        rp1 = (phi_efield - table.phi[0]) / dphi
        ip0 = int(np.floor(rp1) % table.phi.size)
        ip1 = ip0 + 1
        if ip1 == table.phi.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        # fft
        E = Efield.E
        logger.debug(E.x.shape)
        Ex = np.fft.rfft(E.x)
        # frequency [Hz]
        x = fftfreq(Ex.size, Efield.t)
        logger.debug(x.shape)
        # frequency [Hz]
        xp = table.frequency
        ltr = interp(table.leff_theta)  # LWP. m
        lta = interp(np.deg2rad(table.phase_theta))  # LWP. rad
        lpr = interp(table.leff_phi)  # LWP. m
        lpa = interp(np.deg2rad(table.phase_phi))  # LWP. rad
        # Pack the result as a Cartesian vector with complex values
        # fmt: off
        lt = ltr*np.exp( 1j*lta )
        lp = lpr*np.exp( 1j*lpa )
        t_rad, p_rad = np.deg2rad(theta_efield), np.deg2rad(phi_efield)
        ct, st = np.cos(t_rad), np.sin(t_rad)
        cp, sp = np.cos(p_rad), np.sin(p_rad)
        lx = lt*ct*cp - sp*lp
        ly = lt*ct*sp + cp*lp
        lz = -st*lt
        # fmt: on
        # Treating Leff as a vector (no change in magnitude) and transforming
        # it to the shower frame from antenna frame.
        # antenna frame --> ECEF frame --> shower frame
        # TODO: there might be an easier way to do this.
        # vector wrt ECEF frame. Antenna --> ECEF
        a_leff_frame_ant = np.array([lx, ly, lz])
        Leff = np.matmul(self.frame.basis.T, a_leff_frame_ant)
        # vector wrt shower frame. ECEF --> Shower
        Leff = np.matmul(frame.basis, Leff)
        # store effective length
        self.dft_effv_len = Leff
        return Leff

    def compute_voltage(
        self,
        xmax: LTP,
        Efield: ElectricField,
        frame: Union[LTP, GRANDCS, None] = None,
    ) -> Voltage:

        logger.debug("call compute_voltage()")
        # frame is shower frame. self.frame is antenna frame.
        if (self.frame is None) or (frame is None):
            raise MissingFrameError("missing antenna or shower frame")

        # Compute the voltage. input Leff and field are in shower frame.
        def rfft(q):
            return np.fft.rfft(q)

        def irfft(q):
            return np.fft.irfft(q)

        Leff = self.effective_length(xmax, Efield, frame)
        E = Efield.E  # E is CartesianRepresentation
        Ex = rfft(E.x)
        Ey = rfft(E.y)
        Ez = rfft(E.z)
        # Here we have to do an ugly patch for Leff values to be correct
        # fmt: off
        V = irfft( Ex*(Leff[0] - Leff[0,0]) 
                 + Ey*(Leff[1] - Leff[1,0])
                 + Ez*(Leff[2] - Leff[2,0]))
        # fmt: on
        t = Efield.t
        t = t[: V.size]
        return Voltage(t=t, V=V)
