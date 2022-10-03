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
)
from grand.basis.type_trace import ElectricField, Voltage
from grand.io.file_leff import TabulatedAntennaModel

logger = getLogger(__name__)


class MissingFrameError(ValueError):
    pass


@dataclass
class AntennaProcessing:
    model_leff: Any
    frame: Union[LTP, GRANDCS]

    def __post_init__(self):
        assert isinstance(self.model_leff, TabulatedAntennaModel)

    def effective_length(
        self,
        xmax: LTP,
        Efield: ElectricField,
        frame: Union[LTP, GRANDCS],
    ) -> CartesianRepresentation:
        """
        Interpolate effective length in Fourier space by tripler (theta, phi, frequency) linear interpolation
        in direction of the Efield source at antenna position

        :param xmax:
        :param Efield:
        :param frame:
        """

        def fftfreq(size, a_time):
            delta = a_time[1] - a_time[0]
            logger.debug(f"dt_ns = {delta*1e9:.2f}")
            logger.debug(f"{size}")
            return np.fft.fftfreq(size, delta)

        def interp_sphere_freq(val):
            """Linear interpolation (see np.interp() doc) on the frequency axis

            Used to interpolate antenna response in Fourier space defined by modulus and argument
            for 2 axis theta and phi
            => so used 4 times, val is
              * modulus phi
              * argument phi
              * modulus theta
              * argument theta

            The function to be interpolated is itself interpolated by weighting 4 functions
            Interpolation on a sphere => 2 dimension (theta, phi) so 4 positions/values to define interpolation
            positions defined  by index:
              * ip0, ip1 for phi axis
              * it0, it1 for theta axis
            weight used:
              * rp0, rp1 for phi axis, with rp0 + rp1 = 1 [1]
              * rt0, rt1 for theta axis, with rt0 + rt1 = 1 [2]
            """
            # fmt: off
            val_sphere_interpol = rp0 * rt0 * val[:, ip0, it0] \
                     +rp1 * rt0 * val[:, ip1, it0] \
                     +rp0 * rt1 * val[:, ip0, it1] \
                     +rp1 * rt1 * val[:, ip1, it1]
            # fmt: on
            # With [1] and [2]
            # we have: rp0*rt0 + rp1*rt0+ rp0*rt1 + rp1*rt1 = 1
            # so sum of weight is 1, it's also a linear interpolation by angle on sphere
            half_idx = freq_interp_hz.shape[0] // 2
            logger.debug(f"Ref freq: {freq_ref_hz[0]:.3e} {np.max(freq_ref_hz):.3e}")
            logger.debug(f"Interp  : {freq_interp_hz[0]:.3e} {np.max(freq_interp_hz):.3e}")
            val_interp = np.interp(
                freq_interp_hz, freq_ref_hz, val_sphere_interpol, left=0, right=0
            )
            logger.debug(f"val initia : {np.min(val):.3e} {np.max(val):.3e}")
            logger.debug(
                f"Ref    val : {np.min(val_sphere_interpol):.3e} {np.max(val_sphere_interpol):.3e}"
            )
            logger.debug(f"Interp val : {np.min(val_interp):.3e} {np.max(val_interp):.3e}")
            return val_interp

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
        logger.debug(
            f"Source direction (degree): North_gap={float(phi_efield):.1f}, Zenith_dist={float(theta_efield):.1f}"
        )
        # logger.debug(f"{theta_efield.r}")
        # Interpolate using a tri-linear interpolation in (f, phi, theta)
        # table store antenna response
        table = self.model_leff.table
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
        # logger.info(Efield.e_xyz.info())
        E = Efield.e_xyz
        logger.debug(E.x.shape)
        Ex = np.fft.rfft(E.x)
        # frequency [Hz]
        freq_interp_hz = fftfreq(Ex.size, Efield.a_time)
        # frequency [Hz]
        freq_ref_hz = table.frequency
        ltr = interp_sphere_freq(table.leff_theta)  # LWP. m
        lta = interp_sphere_freq(np.deg2rad(table.phase_theta))  # LWP. rad
        lpr = interp_sphere_freq(table.leff_phi)  # LWP. m
        lpa = interp_sphere_freq(np.deg2rad(table.phase_phi))  # LWP. rad
        # Pack the result as a Cartesian vector with complex values
        # fmt: off
        lt = ltr*np.exp(1j * lta)
        lp = lpr*np.exp(1j * lpa)
        t_rad, p_rad = np.deg2rad(theta_efield), np.deg2rad(phi_efield)
        ct, st = np.cos(t_rad), np.sin(t_rad)
        cp, sp = np.cos(p_rad), np.sin(p_rad)
        self.lx = lt*ct*cp - sp*lp
        self.ly = lt*ct*sp + cp*lp
        self.lz = -st*lt
        del lt, lp, ct, st , cp, sp
        # fmt: on
        # Treating Leff as a vector (no change in magnitude) and transforming
        # it to the shower frame from antenna frame.
        # antenna frame --> ECEF frame --> shower frame
        # TODO: there might be an easier way to do this.
        # vector wrt ECEF frame. AntennaProcessing --> ECEF
        a_leff_frame_ant = np.array([self.lx, self.ly, self.lz])
        Leff = np.matmul(self.frame.basis.T, a_leff_frame_ant)
        self.leff_frame_ant = a_leff_frame_ant
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
        # Leff = np.ones((3,500), dtype=np.complex128)
        Leff = self.effective_length(xmax, Efield, frame)
        # logger.info(Leff.dtype)
        # logger.info(Leff.shape)
        E = Efield.e_xyz  # E is CartesianRepresentation
        Ex = np.fft.rfft(E.x)
        Ey = np.fft.rfft(E.y)
        Ez = np.fft.rfft(E.z)
        logger.debug(f"{np.max(E.x)}, {np.max(E.y)}, {np.max(E.z)}")
        # Here we have to do an ugly patch for Leff values to be correct
        # fmt: off
        resp_volt = np.fft.irfft(
                Ex*(Leff[0] - Leff[0, 0]) 
               +Ey*(Leff[1] - Leff[1, 0])
               +Ez*(Leff[2] - Leff[2, 0]))
        # fmt: on
        t = Efield.a_time[: resp_volt.size]
        logger.debug(f"time : {t.dtype} {t.shape}")
        logger.debug(f"volt : {resp_volt.dtype} {resp_volt.shape}")
        return Voltage(t=t, V=resp_volt)

    def compute_voltage_fake(
        self,
        xmax: LTP,
        Efield: ElectricField,
        frame: Union[LTP, GRANDCS, None] = None,
    ) -> Voltage:
        fake = np.ones((998,), dtype=np.float64)
        return Voltage(t=fake, V=fake)
