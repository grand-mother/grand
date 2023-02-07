"""
Processing effective length of antenna
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from logging import getLogger
from typing import Union, Any

import numpy as np
import scipy.fft as sf

from grand.geo.coordinates import (
    LTP,
    GRANDCS,
    CartesianRepresentation,
    SphericalRepresentation,
    ECEF,
)
from grand.basis.type_trace import ElectricField, Voltage
from grand.io.file_leff import TabulatedAntennaModel

logger = getLogger(__name__)


class MissingFrameError(ValueError):
    """Error class"""

@dataclass
class PreComputeInterp:
    # index first in band 30-250MHz
    idx_first : Any=None
    # index last plus one in band 30-250MHz
    idx_lastp1 : Any=None
    # array of index where f_out are in f_sampling
    idx_itp : Any=None
    # array of coefficient inf
    c_inf : Any=None
    # array of coefficient sup
    # c_inf + c_sup = 1
    c_sup : Any=None


@dataclass
class AntennaProcessing:
    """
    Processing effective length of antenna
    """

    model_leff: Any
    pos: Union[LTP, GRANDCS]
    pre_cpt: ClassVar[PreComputeInterp]

    def __post_init__(self):
        assert isinstance(self.model_leff, TabulatedAntennaModel)
        self.size_fft = 0
        self.freqs_out_hz = 0

    @classmethod
    def init_interpolation(cls, freq_out_mhz, freq_sampling_mhz):
        pre = PreComputeInterp()
        d_freq_out = freq_out_mhz[1]
        # + 1 to have first in band
        idx_first = int(freq_sampling_mhz[0] / d_freq_out) + 1
        # + 1 to have first out band => the last plus one 
        idx_lastp1 = int(freq_sampling_mhz[-1] / d_freq_out) + 1
        pre.idx_first = idx_first
        pre.idx_lastp1 = idx_lastp1
        d_freq = freq_sampling_mhz[1] - freq_sampling_mhz[0]
        freq_in_band = freq_out_mhz[idx_first:idx_lastp1]
        pre.idx_itp = np.trunc((freq_in_band - freq_sampling_mhz[0])/d_freq).astype(int)
        pre.c_sup = (freq_in_band - freq_sampling_mhz[pre.idx_itp])/d_freq
        pre.c_inf = 1 - pre.c_sup
        cls.pre_cpt = pre
        logger.info(pre)
        
    def set_out_freq_mhz(self, a_freq):
        """
        !
        typically the return of scipy.fft.rfftfreq/1e6

        :param a_freq:
        :type a_freq:
        """
        assert isinstance(a_freq, np.ndarray)
        assert a_freq[0] == 0
        self.freqs_out_hz = a_freq * 1e6
        # we used rfreqfft
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html
        self.size_fft = 2 * (a_freq.shape[0] - 1)
        logger.debug(f"size_fft: {self.size_fft}")
        

    def effective_length_slow(
        self,
        xmax: LTP,
        efield: ElectricField,
        frame: Union[LTP, GRANDCS],
    ) -> CartesianRepresentation:
        """
        Calculation of FFT length effective of antenna for a given Xmax position.

        Linear interpolation of template of effective length in Fourier space
        by tripler (theta, phi, frequency)
        in direction of the efield source at antenna position

        :param xmax:
        :param efield:
        :param frame:
        """

        def fftfreq(size, a_time):
            delta = a_time[1] - a_time[0]
            logger.debug(f"dt_ns = {delta*1e9:.2f}")
            logger.debug(f"{size}")
            return np.fft.rfftfreq(size, delta)
        
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
            Interpolation on a sphere => 2 dimension (theta, phi) so 4 positions/values to
            define interpolation positions defined  by index:
              * ip0, ip1 for phi axis
              * it0, it1 for theta axis
            weight used:
              * rp0, rp1 for phi axis, with rp0 + rp1 = 1 [1]
              * rt0, rt1 for theta axis, with rt0 + rt1 = 1 [2]
            """
            logger.debug(f"val shape: {val.shape}")
            logger.debug(f"rp0 , rt0 : {rp0} {rt0}")
            # fmt: off
            val_sphere_interpol = rp0 * rt0 * val[:, ip0, it0] \
                     +rp1 * rt0 * val[:, ip1, it0] \
                     +rp0 * rt1 * val[:, ip0, it1] \
                     +rp1 * rt1 * val[:, ip1, it1]
            # fmt: on
            # With [1] and [2]
            # we have: rp0*rt0 + rp1*rt0+ rp0*rt1 + rp1*rt1 = 1
            # so sum of weight is 1, it's also a linear interpolation by angle on sphere
            logger.debug(f"Ref freq: {freq_ref_hz[0]:.3e} {np.max(freq_ref_hz):.3e}")
            logger.debug(f"Interp  : {self.freqs_out_hz[0]:.3e} {np.max(self.freqs_out_hz):.3e}")
            if False:
                val_interp = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
                val_interp_band = np.interp(
                    self.freqs_out_hz[self.idx_freq_min : self.idx_freq_max + 1],
                    freq_ref_hz,
                    val_sphere_interpol,
                    left=0,
                    right=0,
                )
                val_interp[self.idx_freq_min : self.idx_freq_max + 1] = val_interp_band
            else:
                val_interp = np.interp(
                    self.freqs_out_hz, freq_ref_hz, val_sphere_interpol, left=0, right=0
                )
            # NOTE: scipy interp1d isn'r better than numpy interp
            logger.debug(f"val initia : {np.min(val):.3e} {np.max(val):.3e}")
            logger.debug(
                f"Ref    val : {np.min(val_sphere_interpol):.3e} {np.max(val_sphere_interpol):.3e}"
            )
            logger.debug(f"Interp val : {np.min(val_interp):.3e} {np.max(val_interp):.3e}")
            return val_interp

        # 'frame' is shower frame. 'self.frame' is antenna frame.
        if isinstance(xmax, LTP):
            # shower frame --> antenna frame
            # TODO: why next takes time ? => LPT init 35 ms ????
            # direction = xmax.ltp_to_ltp(self.pos)
            ecef = ECEF(xmax)
            pos_v = np.array(
                (
                    ecef.x - self.pos.location.x,
                    ecef.y - self.pos.location.y,
                    ecef.z - self.pos.location.z,
                )
            )
            direction = np.matmul(self.pos.basis, pos_v)
        else:
            raise TypeError(f"Provide Xmax in LTP frame instead of {type(xmax)}")
        # compute efield direction in spherical coordinate in antenna frame
        direction_cart = CartesianRepresentation(x=direction[0], y=direction[1], z=direction[2])
        direction_sphr = SphericalRepresentation(direction_cart)
        theta_efield, phi_efield = direction_sphr.theta, direction_sphr.phi
        logger.debug(f"type theta_efield: {type(theta_efield)} {theta_efield}")
        logger.info(
            f"Source direction (degree): North_gap={float(phi_efield):.1f},\
             Zenith_dist={float(theta_efield):.1f}"
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
        # phi_efield between index ip0 and ip1 in phi antenna response representation
        dphi = table.phi[1] - table.phi[0]  # deg
        rp1 = (phi_efield - table.phi[0]) / dphi
        ip0 = int(np.floor(rp1) % table.phi.size)
        ip1 = ip0 + 1
        if ip1 == table.phi.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        # fft
        # logger.info(efield.e_xyz.info())
        e_xyz = efield.e_xyz
        logger.debug(e_xyz.shape)
        # fft_ex = np.fft.rfft(E.x)
        # frequency [Hz] with padding
        if self.size_fft == 0:
            self.size_fft = sf.next_fast_len(e_xyz.shape[1])
            self.freqs_out_hz = fftfreq(self.size_fft, efield.a_time)
        logger.debug(f"size_fft={self.size_fft}")
        # Leff ref frequency  [Hz]
        freq_ref_hz = table.frequency
        # interpolation Leff theta and phi
        logger.debug(f"leff_phi_cart: {table.leff_phi_cart.shape}")
        # l_t = interp_sphere_freq(table.leff_theta_cart)
        # l_p = interp_sphere_freq(table.leff_phi_cart)
        ltr = interp_sphere_freq(table.leff_theta_cart.real)
        lpr = interp_sphere_freq(table.leff_phi_cart.real)
        lti = interp_sphere_freq(table.leff_theta_cart.imag)
        lpi = interp_sphere_freq(table.leff_phi_cart.imag)
        # Pack the result as a Cartesian vector with complex values
        l_t = ltr + 1j * lti
        l_p = lpr + 1j * lpi
        # fmt: off
        t_rad, p_rad = np.deg2rad(theta_efield), np.deg2rad(phi_efield)
        c_t, s_t = np.cos(t_rad), np.sin(t_rad)
        c_p, s_p = np.cos(p_rad), np.sin(p_rad)
        self.l_x = l_t * c_t * c_p - s_p * l_p
        self.l_y = l_t * c_t * s_p + c_p * l_p
        self.l_z = -s_t * l_t
        # del l_t, l_p, c_t, s_t , c_p, s_p
        # fmt: on
        # Treating Leff as a vector (no change in magnitude) and transforming
        # it to the shower frame from antenna frame.
        # antenna frame --> ECEF frame --> shower frame
        # TODO: there might be an easier way to do this.
        # vector wrt ECEF frame. AntennaProcessing --> ECEF
        fft_leff_frame_ant = np.array([self.l_x, self.l_y, self.l_z])
        fft_leff = np.matmul(self.pos.basis.T, fft_leff_frame_ant)
        self.leff_frame_ant = fft_leff_frame_ant
        # vector wrt shower frame. ECEF --> Shower
        self.fft_leff_frame_shower = np.matmul(frame.basis, fft_leff)
        return self.fft_leff_frame_shower
    
    
    def effective_length(
        self,
        xmax: LTP,
        efield: ElectricField,
        frame: Union[LTP, GRANDCS],
    ) -> CartesianRepresentation:
        """
        Calculation of FFT length effective of antenna for a given Xmax position.

        Linear interpolation of template of effective length in Fourier space
        by tripler (theta, phi, frequency)
        in direction of the efield source at antenna position

        :param xmax:
        :param efield:
        :param frame:
        """
        # 'frame' is shower frame. 'self.frame' is antenna frame.
        if isinstance(xmax, LTP):
            # shower frame --> antenna frame
            # TODO: why next takes time ? => LPT init 35 ms ????
            # direction = xmax.ltp_to_ltp(self.pos)
            ecef = ECEF(xmax)
            pos_v = np.array(
                (
                    ecef.x - self.pos.location.x,
                    ecef.y - self.pos.location.y,
                    ecef.z - self.pos.location.z,
                )
            )
            direction = np.matmul(self.pos.basis, pos_v)
        else:
            raise TypeError(f"Provide Xmax in LTP frame instead of {type(xmax)}")
        # compute efield direction in spherical coordinate in antenna frame
        direction_cart = CartesianRepresentation(x=direction[0], y=direction[1], z=direction[2])
        direction_sphr = SphericalRepresentation(direction_cart)
        theta_efield, phi_efield = direction_sphr.theta, direction_sphr.phi
        logger.debug(f"type theta_efield: {type(theta_efield)} {theta_efield}")
        logger.info(
            f"Source direction (degree): North_gap={float(phi_efield):.1f},\
             Zenith_dist={float(theta_efield):.1f}"
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
        # phi_efield between index ip0 and ip1 in phi antenna response representation
        dphi = table.phi[1] - table.phi[0]  # deg
        rp1 = (phi_efield - table.phi[0]) / dphi
        ip0 = int(np.floor(rp1) % table.phi.size)
        ip1 = ip0 + 1
        if ip1 == table.phi.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        # fft
        # logger.info(efield.e_xyz.info())
        e_xyz = efield.e_xyz
        logger.debug(e_xyz.shape)
        # fft_ex = np.fft.rfft(E.x)
        # frequency [Hz] with padding
        if self.size_fft == 0:
            # must be fix to use precompute interpolation
            raise
        logger.debug(f"size_fft={self.size_fft}")
        # Leff ref frequency  [Hz]
        freq_ref_hz = table.frequency
        # interpolation Leff theta and phi on sphere
        logger.debug(f"leff_phi_cart: {table.leff_phi_cart.shape}")
        leff = table.leff_theta_cart
        leff_itp_t = rp0 * rt0 * leff[ ip0, it0, :] \
            +rp1 * rt0 * leff[ip1, it0, :] \
            +rp0 * rt1 * leff[ip0, it1, :] \
            +rp1 * rt1 * leff[ip1, it1, :]
        leff = table.leff_phi_cart
        leff_itp_p = rp0 * rt0 * leff[ ip0, it0, :] \
            +rp1 * rt0 * leff[ip1, it0, :] \
            +rp0 * rt1 * leff[ip0, it1, :] \
            +rp1 * rt1 * leff[ip1, it1, :]
        leff_itp_sph = np.array([leff_itp_t, leff_itp_p])
        logger.info(leff_itp_sph.shape)
        # interpolation Leff theta and phi on frequency
        pre = AntennaProcessing.pre_cpt
        logger.info(pre.c_inf.shape)
        logger.info(pre.idx_itp.shape)
        logger.info(leff_itp_sph[:,pre.idx_itp].shape)
        leff_itp  = pre.c_inf*leff_itp_sph[:,pre.idx_itp] + pre.c_sup*leff_itp_sph[:,pre.idx_itp + 1]
        # now add zeros outside leff frequency band and unpack leff theta , phi 
        l_t = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_t[pre.idx_first:pre.idx_lastp1] = leff_itp[0]
        l_p = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_p[pre.idx_first:pre.idx_lastp1] = leff_itp[1]
        # fmt: off
        t_rad, p_rad = np.deg2rad(theta_efield), np.deg2rad(phi_efield)
        c_t, s_t = np.cos(t_rad), np.sin(t_rad)
        c_p, s_p = np.cos(p_rad), np.sin(p_rad)
        self.l_x = l_t * c_t * c_p - s_p * l_p
        self.l_y = l_t * c_t * s_p + c_p * l_p
        self.l_z = -s_t * l_t
        # fmt: on
        #del l_t, l_p, c_t, s_t , c_p, s_p
        # Treating Leff as a vector (no change in magnitude) and transforming
        # it to the shower frame from antenna frame.
        # antenna frame --> ECEF frame --> shower frame
        # TODO: there might be an easier way to do this.
        # vector wrt ECEF frame. AntennaProcessing --> ECEF
        fft_leff_frame_ant = np.array([self.l_x, self.l_y, self.l_z])
        fft_leff = np.matmul(self.pos.basis.T, fft_leff_frame_ant)
        self.leff_frame_ant = fft_leff_frame_ant
        # vector wrt shower frame. ECEF --> Shower
        self.fft_leff_frame_shower = np.matmul(frame.basis, fft_leff)
        return self.fft_leff_frame_shower

    def compute_voltage(
        self,
        xmax: LTP,
        efield: ElectricField,
        frame: Union[LTP, GRANDCS, None] = None,
    ) -> Voltage:
        """

        :param xmax:
        :type xmax:
        :param efield:
        :type efield:
        :param frame:
        :type frame:
        """
        # frame is shower frame. self.frame is antenna frame.
        if (self.pos is None) or (frame is None):
            raise MissingFrameError("missing antenna or shower frame")

        # Compute the voltage. input fft_leff and field are in shower frame.
        fft_leff = self.effective_length(xmax, efield, frame)
        logger.debug(fft_leff.shape)
        # E is CartesianRepresentation
        fft_e = efield.get_fft(self.size_fft)
        logger.debug(f"size_fft leff={fft_leff.shape}")
        logger.debug(f"size_fft efield={fft_e.shape}")
        # convol e_xyz field by Leff in Fourier space
        self.fft_resp_volt = (
            fft_e[0] * fft_leff[0] + fft_e[1] * fft_leff[1] + fft_e[2] * fft_leff[2]
        )
        # inverse FFT and remove zero-padding
        resp_volt = sf.irfft(self.fft_resp_volt)[: efield.e_xyz.shape[1]]
        # WARNING do not used : sf.irfft(self.fft_resp_volt, efield.e_xyz.shape[1])
        t = efield.a_time
        logger.debug(f"time : {t.dtype} {t.shape}")
        logger.debug(f"volt : {resp_volt.dtype} {resp_volt.shape}")
        return Voltage(t=t, V=resp_volt)
