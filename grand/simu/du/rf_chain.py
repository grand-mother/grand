"""!
Simulation of the effects detector on the signal : VSWR, LNA, cable, 

Reference document : 
  * "RF Chain simulation for GP300" by Xidian University, 
    Pengfei Zhang and 
    Pengxiong Ma, Chao Zhang, Rongjuan Wand, Xin Xu
    
  * Article 

Reference code: 
  * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py
"""

import os.path
import math
from logging import getLogger

from numpy.ma import log10, abs
from scipy import interpolate
import scipy.fft as sf
import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data_model

logger = getLogger(__name__)


def func_interpol(a_x, a_y):
    """!
    function of interpolation, set to zero outside interval definition a_x
    """
    assert a_x.shape[0] > 0
    return interpolate.interp1d(a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0))


class GenericProcessingDU:
    """ """

    def __init__(self):
        """ """
        self.freqs_out = np.zeros(0)
        self.size_out = 0

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        # fix a file version for processing by heritage
        raise

    ### SETTER

    def set_out_freq_mhz(self, a_freq):
        """!
        typically the return of scipy.fft.rfftfreq/1e6
        """
        assert isinstance(a_freq, np.ndarray)
        assert a_freq[0] == 0
        self.freqs_out = a_freq
        self.size_out = a_freq.shape[0]
        self.size_sig = (self.size_out - 1) * 2


class StandingWaveRatio(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaption Colley jm

    Class goals:
      * define VSWR value as s11 parameter
      * read only once data files
      * pre_compute interpolation
    """

    def __init__(self):
        super().__init__()
        self.s11 = np.zeros((0, 3), dtype=np.complex64)

    def compute_s11(self, unit=1):
        """!

        @param unit:
        """
        s11 = np.zeros((self.size_out, 3), dtype=np.complex64)
        for axis in range(3):
            filename = self._set_name_data_file(axis)
            freq = np.loadtxt(filename, usecols=0) / 1e6  # HZ to MHz
            if unit == 0:
                re = np.loadtxt(filename, usecols=1)
                im = np.loadtxt(filename, usecols=2)
                db = 20 * log10(abs(re + 1j * im))
            elif unit == 1:
                db = np.loadtxt(filename, usecols=1)
                deg = np.loadtxt(filename, usecols=2)
                mag = 10 ** (db / 20)
                re = mag * np.cos(deg / 180 * math.pi)
                im = mag * np.sin(deg / 180 * math.pi)
            if axis == 0:
                db_s11 = np.zeros((3, len(freq)))
            db_s11[axis] = db
            #
            f_re = func_interpol(freq, re)
            renew = f_re(self.freqs_out)
            f_im = func_interpol(freq, im)
            imnew = f_im(self.freqs_out)
            s11[:, axis] = renew + 1j * imnew
        self.s11 = s11
        self.f_name = filename
        self._db_s11 = db_s11
        self._f_db_s11 = freq

    def plot_vswr(self):  # pragma: no cover
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"VSWR, s11 parameter")
        ax1.set_title("db")
        ax1.plot(self._f_db_s11, self._db_s11[0], label="0")
        ax1.plot(self._f_db_s11, self._db_s11[1], label="1")
        ax1.plot(self._f_db_s11, self._db_s11[2], label="2")
        ax1.set_ylabel(f"db")
        ax1.set_xlabel(f"[MHz]")
        ax1.grid()
        ax1.legend()
        ax2.set_title("abs(s_11)")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 0]), label="0")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 1]), label="1")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 2]), label="2")
        ax2.set_ylabel(f"?")
        ax2.set_xlabel(f"[MHz]")
        ax2.grid()
        ax2.legend()


class StandingWaveRatioGP300(StandingWaveRatio):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaptation Colley jm

    Class goals:
      * define data model used for GP300
    """

    def __init__(self):
        super().__init__()

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        lna_address = os.path.join("detector", "antennaVSWR", f"{axis+1}.s1p")
        return grand_add_path_data_model(lna_address)


class LowNoiseAmplificator(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaption Colley jm

    Class goals:
      * Perform the LNA filter on signal for each antenna of GP300
      * read only once LNA data files
      * pre_compute interpolation
    """

    def __init__(self):
        """

        @param size_sig: size of the trace after
        """
        super().__init__()
        self.data_lna = []
        for axis in range(3):
            lna = np.loadtxt(self._set_name_data_file(axis))
            # Hz to MHz
            lna[0] /= 1e6
            self.data_lna.append(lna)
        self.f_lna = lna[:, 0]

    ### INTERNAL

    def set_vswr_model(self, o_vswr):
        """Set VSWR model and upadte value if transfert function Rho

        @param o_vswr (StandingWaveRatio):
        """
        assert isinstance(o_vswr, StandingWaveRatio)
        self._vswr = o_vswr

    def _pre_compute(self, unit=1):
        """!
        compute what is possible without know response antenna

        @param unit: select LNA type stockage 0: [re, im],  1: [amp, arg]
        """
        lna_gama = np.zeros((self.size_out, 3), dtype=np.complex64)
        lna_s21 = np.zeros((self.size_out, 3), dtype=np.complex64)
        dbs21_a = np.zeros((3, self.f_lna.size))
        logger.debug(f"{dbs21_a.shape}")
        logger.debug(f"{self.data_lna[0].shape}")
        for axis in range(3):
            freq = self.data_lna[axis][:, 0] / 1e6  # HZ to MHz
            if unit == 0:
                res11 = self.data_lna[axis][:, 1]
                ims11 = self.data_lna[axis][:, 2]
                res21 = self.data_lna[axis][:, 3]
                ims21 = self.data_lna[axis][:, 4]
                dbs21 = 20 * log10(abs(res21 + 1j * ims21))
            elif unit == 1:
                # TODO: unit 0 and 1 used same index !!
                # see code reference
                dbs11 = self.data_lna[axis][:, 1]
                degs11 = self.data_lna[axis][:, 2]
                mags11 = 10 ** (dbs11 / 20)
                res11 = mags11 * np.cos(np.deg2rad(degs11))
                ims11 = mags11 * np.sin(np.deg2rad(degs11))
                dbs21 = self.data_lna[axis][:, 3]
                degs21 = self.data_lna[axis][:, 4]
                mags21 = 10 ** (dbs21 / 20)
                res21 = mags21 * np.cos(np.deg2rad(degs21))
                ims21 = mags21 * np.sin(np.deg2rad(degs21))
            dbs21_a[axis] = dbs21
            #
            f_res11 = func_interpol(freq, res11)
            res11new = f_res11(self.freqs_out)
            f_ims11 = func_interpol(freq, ims11)
            ims11new = f_ims11(self.freqs_out)
            lna_gama[:, axis] = res11new + 1j * ims11new
            #
            f_res21 = func_interpol(freq, res21)
            res21new = f_res21(self.freqs_out)
            f_ims21 = func_interpol(freq, ims21)
            ims21new = f_ims21(self.freqs_out)
            lna_s21[:, axis] = res21new + 1j * ims21new
        # Add attribut
        self.lna_gama = lna_gama
        self.lna_s21 = lna_s21
        self._dbs21_a = dbs21_a

    def _compute(self, antenna_gama):
        """!update_with_s11

        @param antenna_gama (N,3):
        """
        assert antenna_gama.shape[0] > 0
        assert antenna_gama.shape[1] == 3
        self._pre_compute()
        z_0 = 50
        z_ant = z_0 * (1 + antenna_gama) / (1 - antenna_gama)
        z_in_lna = z_0 * (1 + self.lna_gama) / (1 - self.lna_gama)
        print(z_ant.shape)
        print(z_in_lna.shape)
        self._rho1 = z_in_lna / (z_ant + z_in_lna)
        self._rho2 = (1 + self.lna_gama) / (1 - antenna_gama * self.lna_gama)
        self._rho3 = self.lna_s21 / (1 + self.lna_gama)
        self.rho123 = self._rho1 * self._rho2 * self._rho3
        self.z_ant = z_ant
        self.z_in_lna = z_in_lna
        self.antenna_gama = antenna_gama

    ### OPERATION

    def compute_at_freqs(self, a_freq_mhz):
        """!compute Rho transfert function for frequency  a_freq

        @param a_freq: vector of frequency
        """
        super().set_out_freq_mhz(a_freq_mhz)
        self._vswr.set_out_freq_mhz(a_freq_mhz)
        self._vswr.compute_s11()
        # update rho
        self._compute(self._vswr.s11)

    ### GETTER

    def get_rho(self):
        return self.rho123.T

    ### PLOT

    def plot_gama(self):  # pragma: no cover
        plt.figure()
        plt.title("antenna_gama")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 0]), label=r"0")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 1]), label=r"1")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 2]), label=r"2")
        # plt.plot(self.freqs_out, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_z(self):  # pragma: no cover
        plt.figure()
        plt.title("")
        plt.plot(self.freqs_out, np.abs(self.z_ant[:, 0]), label=r"$\mathregular{Z_{A}}$")
        plt.plot(self.freqs_out, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_lna(self):  # pragma: no cover
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(self.f_lna, self._dbs21_a[p])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.figure(figsize=(9, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for p in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho1[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.subplot(1, 3, 2)
        for p in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho2[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.subplot(1, 3, 3)
        for p in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho3[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    def plot_rho_kernel(self):
        plt.figure()
        plt.title("Rho kernel")
        kernel_rho = sf.ifftshift(sf.irfft(self.get_rho()))
        print(kernel_rho.shape)
        # TODO: self.size_sig//2 or self.size_sig//2 -1 ?
        v_time = np.arange(self.size_sig, dtype=np.float64) - self.size_sig // 2
        dt_ns = 1e9 / (self.freqs_out[1] * self.size_sig * 1e6)
        v_time_ns = dt_ns * v_time
        plt.plot(v_time_ns, kernel_rho[0], label="0")
        plt.plot(v_time_ns, kernel_rho[1], label="1")
        plt.plot(v_time_ns, kernel_rho[2], label="2")
        plt.xlabel("ns")
        plt.grid()
        plt.legend()


class LowNoiseAmplificatorGP300(LowNoiseAmplificator):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaption Colley jm

    Class goals:
      * define data model used for GP300
    """

    def __init__(self):
        super().__init__()
        self._vswr = StandingWaveRatioGP300()

    ### INTERNAL

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        lna_address = os.path.join("detector", "LNASparameter", f"{axis+1}.s2p")
        return grand_add_path_data_model(lna_address)


class MasterRfChain:
    """
    Facade for all detector effect
    """

    def __init__(self):
        pass
