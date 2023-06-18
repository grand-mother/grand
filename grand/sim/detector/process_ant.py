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
#from grand.io.file_leff import TabulatedAntennaModel

logger = getLogger(__name__)


class MissingFrameError(ValueError):
    """Error class"""


@dataclass
class PreComputeInterpol:
    """
    Precompute linear interpolation of frequency of Leff
    """

    # index of freq in first in band 30-250MHz
    idx_first: Any = None
    # index of freq in last plus one in band 30-250MHz
    idx_lastp1: Any = None # JM
    #idx_last: Any = None    # RK: replaced idx_lastp1 to idx_last
    # array of index where f_out are in f_in for 
    # interpolation between idx_itp[i], idx_itp[i]+1
    # with c_inf and c_sup coefficient
    idx_itp: Any = None
    # array of coefficient inf
    c_inf: Any = None
    # array of coefficient sup
    # c_inf + c_sup = 1
    c_sup: Any = None

    def init_linear_interpol(self, freq_in_mhz, freq_out_mhz):
        """
        Precompute coefficient of linear interpolation for freq_out_mhz with reference defined at freq_in_mhz
        :param freq_in_mhz: regular array of frequency where function is defined
        :param freq_out_mhz: regular array frequency where we want interpol
        """
        d_freq_out = freq_out_mhz[1]
        # index of freq in first in band, + 1 to have first in band
        idx_first = int(freq_in_mhz[0] / d_freq_out) + 1
        # index of freq in last plus one, + 1 to have first out band
        idx_lastp1 = int(freq_in_mhz[-1] / d_freq_out) + 1
        # https://github.com/grand-mother/collaboration-issues/issues/30
        #idx_last = int(freq_in_mhz[-1] / d_freq_out)      # RK: idx_lastp1 --> idx_last
        self.idx_first = idx_first
        self.idx_lastp1 = idx_lastp1
        d_freq_in = freq_in_mhz[1] - freq_in_mhz[0]
        freq_in_band = freq_out_mhz[idx_first:idx_lastp1]
        self.idx_itp = np.trunc((freq_in_band - freq_in_mhz[0]) / d_freq_in).astype(int)
        # define coefficient of linear interpolation
        self.c_sup = (freq_in_band - freq_in_mhz[self.idx_itp]) / d_freq_in
        if self.idx_itp[-1]+1 == freq_in_mhz.shape[0]:
            # https://github.com/grand-mother/collaboration-issues/issues/30
            logger.info(f" ** Specfic processing when f_in = k * f_out else IndexError **")
            self.idx_itp[-1] -= 1
            # in this case last c_sup must be zero
            # check it !
            assert np.allclose(self.c_sup[-1], 0)
        self.c_inf = 1 - self.c_sup

    def get_linear_interpol(self, a_val):
        """
        Return f(freq_out_mhz) by linear interpolation of f defined by
        f(freq_in_mhz) = a_val
        :param a_val: defined value of function at freq_in_mhz
        """
        a_itp = self.c_inf * a_val[self.idx_itp] + self.c_sup * a_val[self.idx_itp + 1]
        return a_itp


@dataclass
class AntennaProcessing:
    """
    Processing effective length of antenna
    """

    model_leff: Any
    pos: Union[LTP, GRANDCS]
    pre_cpt: ClassVar[PreComputeInterpol]

    def __post_init__(self):
        #assert isinstance(self.model_leff, TabulatedAntennaModel)
        self.size_fft = 0
        self.freqs_out_hz = 0

    @classmethod
    def init_interpolation(cls, freq_sampling_mhz, freq_out_mhz):
        pre = PreComputeInterpol()
        pre.init_linear_interpol(freq_sampling_mhz, freq_out_mhz)
        cls.pre_cpt = pre
        logger.debug(pre)

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
        # delta theta in degree
        dtheta = self.model_leff.theta[1] - self.model_leff.theta[0]
        # theta_efield between index it0 and it1 in theta antenna response representation
        rt1 = (theta_efield - self.model_leff.theta[0]) / dtheta
        # prevent > 360 deg or >180 deg ?
        it0 = int(np.floor(rt1) % self.model_leff.theta.size)
        it1 = it0 + 1
        if it1 == self.model_leff.theta.size:  # Prevent overflow
            it1, rt1 = it0, 0
        else:
            rt1 -= np.floor(rt1)
        rt0 = 1 - rt1
        # phi_efield between index ip0 and ip1 in phi antenna response representation
        dphi = self.model_leff.phi[1] - self.model_leff.phi[0]  # deg
        rp1 = (phi_efield - self.model_leff.phi[0]) / dphi
        ip0 = int(np.floor(rp1) % self.model_leff.phi.size)
        ip1 = ip0 + 1
        if ip1 == self.model_leff.phi.size:  # Results are periodic along phi
            ip1 = 0
        rp1 -= np.floor(rp1)
        rp0 = 1 - rp1
        # fft
        # logger.info(efield.e_xyz.info())
        e_xyz = efield.e_xyz
        logger.debug(e_xyz.shape)
        # Leff ref frequency  [Hz]
        freq_ref_hz = self.model_leff.frequency
        # frequency [Hz] with padding
        if self.size_fft == 0:
            # case without init before call effective_length
            logger.warning(
                "Run AntennaProcessing.init_interpolation() only once per event.\
                Define output frequency before calling effective_length by calling set_out_freq_mhz()"
            )
            freq_out = sf.rfftfreq(efield.get_nb_sample(), efield.get_delta_time_s()) * 1e-6
            self.set_out_freq_mhz(freq_out)
            # frequency in MHz
            AntennaProcessing.init_interpolation(freq_ref_hz * 1e-6, freq_out)

        logger.debug(f"size_fft={self.size_fft}")
        # interpolation Leff theta and phi on sphere
        leff = self.model_leff.leff_theta_reim
        leff_itp_t = (
                 rp0 * rt0 * leff[:, ip0, it0] \
                +rp1 * rt0 * leff[:, ip1, it0] \
                +rp0 * rt1 * leff[:, ip0, it1] \
                +rp1 * rt1 * leff[:, ip1, it1]
                )
        leff = self.model_leff.leff_phi_reim
        leff_itp_p = (
                 rp0 * rt0 * leff[:, ip0, it0] \
                +rp1 * rt0 * leff[:, ip1, it0] \
                +rp0 * rt1 * leff[:, ip0, it1] \
                +rp1 * rt1 * leff[:, ip1, it1]
                )
        leff_itp_sph = np.array([leff_itp_t, leff_itp_p])
        # interpolation Leff theta and phi on frequency
        pre = AntennaProcessing.pre_cpt
        leff_itp = (
            pre.c_inf * leff_itp_sph[:, pre.idx_itp] + pre.c_sup * leff_itp_sph[:, pre.idx_itp + 1]
        )
        # now add zeros outside leff frequency band and unpack leff theta , phi
        l_t = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_t[pre.idx_first : pre.idx_lastp1] = leff_itp[0]
        l_p = np.zeros(self.freqs_out_hz.shape[0], dtype=np.complex64)
        l_p[pre.idx_first : pre.idx_lastp1] = leff_itp[1]
        # fmt: off
        t_rad, p_rad = np.deg2rad(theta_efield), np.deg2rad(phi_efield)
        c_t, s_t = np.cos(t_rad), np.sin(t_rad)
        c_p, s_p = np.cos(p_rad), np.sin(p_rad)
        self.l_x = l_t * c_t * c_p - s_p * l_p
        self.l_y = l_t * c_t * s_p + c_p * l_p
        self.l_z = -s_t * l_t
        # fmt: on
        # del l_t, l_p, c_t, s_t , c_p, s_p
        # Treating Leff as a vector (no change in magnitude) and transforming
        # it to the shower frame from antenna frame.
        # antenna frame --> ECEF frame --> shower frame
        # TODO: there might be an easier way to do this.
        # vector wrt ECEF frame. AntennaProcessing --> ECEF
        self.leff_frame_ant = np.array([self.l_x, self.l_y, self.l_z])
        leff_frame_ecef = np.matmul(self.pos.basis.T, self.leff_frame_ant)
        # vector wrt shower frame. ECEF --> Shower
        self.leff_frame_shower = np.matmul(frame.basis, leff_frame_ecef)

        return self.leff_frame_shower

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