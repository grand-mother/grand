"""!
Simulation of the effects detector on the signal : VSWR, LNA, cable, VGA ,Balun, filter

@authors PengFei Zhang and Xidian group, GRANDLIB adaptation Colley JM

Reference document : 
  * "RF Chain simulation for GP300" by Xidian University, 
    Pengfei Zhang and 
    Pengxiong Ma, Chao Zhang, Rongjuan Wand, Xin Xu
    
  * Article 

Reference code: 
  * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py
"""

import os.path
from logging import getLogger

from scipy import interpolate
import scipy.fft as sf
import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data_model

logger = getLogger(__name__)


def interpol_at_new_x(a_x, a_y, new_x):
    """!
    Interpolation of discreet function F defined by a set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    @param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    @param a_y (float, (N)): F(a_x) = a_y
    @param new_x (float, (M)): new value of x

    @return F(new_x) (float, (M)): interpolation if F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)


class GenericProcessingDU:
    """ """

    def __init__(self):
        """ """
        self.freqs_out = np.zeros(0)
        self.nb_freqs = 0
        self.size_sig = 0

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        # fix a file version for processing by heritage
        pass

    ### SETTER

    def set_out_freq_mhz(self, a_freq):
        """Define frequencies

        @param a_freq (float, (N)):[MHz] given by scipy.fft.rfftfreq/1e6
        """
        assert isinstance(a_freq, np.ndarray)
        assert a_freq[0] == 0
        self.freqs_out = a_freq
        self.nb_freqs = a_freq.shape[0]
        self.size_sig = (self.nb_freqs - 1) * 2


class StandingWaveRatioGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group

    Class goals:
      * define VSWR value as s11 parameter
      * read only once data files
      * pre_compute interpolation
    """

    def __init__(self):
        super().__init__()
        self.s11 = np.zeros((0, 3), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        file_address = os.path.join("detector", "antennaVSWR", f"{axis+1}.s1p")
        return grand_add_path_data_model(file_address)

    def compute_s11(self, unit=1):
        """!

        @param unit:
        """
        s11 = np.zeros((self.nb_freqs, 3), dtype=np.complex64)
        for axis in range(3):
            filename = self._set_name_data_file(axis)
            freq = np.loadtxt(filename, usecols=0) / 1e6  # HZ to MHz
            dbel = np.loadtxt(filename, usecols=1)
            deg = np.loadtxt(filename, usecols=2)
            mag = 10 ** (dbel / 20)
            p_re = mag * np.cos(np.deg2rad(deg))
            p_im = mag * np.sin(np.deg2rad(deg))
            if axis == 0:
                db_s11 = np.zeros((3, len(freq)))
            db_s11[axis] = dbel
            #
            s11_real = interpol_at_new_x(freq, p_re, self.freqs_out)
            s11[:, axis] = s11_real + 1j * interpol_at_new_x(freq, p_im, self.freqs_out)
        self.s11 = s11
        self._db_s11 = db_s11
        self._f_db_s11 = freq

    def plot_vswr(self):  # pragma: no cover
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"VSWR, s11 parameter")
        ax1.set_title("db")
        ax1.plot(self._f_db_s11, self._db_s11[0], "k", label="0")
        ax1.plot(self._f_db_s11, self._db_s11[1], "y", label="1")
        ax1.plot(self._f_db_s11, self._db_s11[2], "b", label="2")
        ax1.set_ylabel(f"db")
        ax1.set_xlabel(f"[MHz]")
        ax1.grid()
        ax1.legend()
        ax2.set_title("abs(s_11)")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 0]), "k", label="0")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 1]), "y", label="1")
        ax2.plot(self.freqs_out, np.abs(self.s11[:, 2]), "b", label="2")
        ax2.set_ylabel(f"?")
        ax2.set_xlabel(f"[MHz]")
        ax2.grid()
        ax2.legend()


class LowNoiseAmplificatorGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group
    Class goals:
      * Perform the LNA filter on signal for each antenna
      * manage VSWR model
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
            self.data_lna.append(lna)
        self.freqs_in = lna[:, 0] / 1e6
        self._vswr = StandingWaveRatioGP300()
        self.lna_gama = np.zeros((self.nb_freqs, 3), dtype=np.complex64)
        self.lna_s21 = np.zeros((self.nb_freqs, 3), dtype=np.complex64)
        self._dbs21_a = np.zeros((self.nb_freqs, 3), dtype=np.complex64)

    ### INTERNAL

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        lna_address = os.path.join("detector", "LNASparameter", f"{axis+1}.s2p")
        return grand_add_path_data_model(lna_address)

    def _pre_compute(self):
        """!
        compute what is possible without know response antenna

        @param unit: select LNA type stockage 0: [re, im],  1: [amp, arg]
        """
        lna_gama = np.zeros((self.nb_freqs, 3), dtype=np.complex64)
        lna_s21 = np.zeros((self.nb_freqs, 3), dtype=np.complex64)
        dbs21_a = np.zeros((3, self.freqs_in.size))
        logger.debug(f"{dbs21_a.shape}")
        logger.debug(f"{self.data_lna[0].shape}")
        freqs_in = self.freqs_in
        for axis in range(3):
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
            lna_gama[:, axis] = interpol_at_new_x(freqs_in, res11, self.freqs_out)
            lna_gama[:, axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_out)
            #
            lna_s21[:, axis] = interpol_at_new_x(freqs_in, res21, self.freqs_out)
            lna_s21[:, axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_out)
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
        self._rho1 = z_in_lna / (z_ant + z_in_lna)
        self._rho2 = (1 + self.lna_gama) / (1 - antenna_gama * self.lna_gama)
        self._rho3 = self.lna_s21 / (1 + self.lna_gama)
        self.rho123 = self._rho1 * self._rho2 * self._rho3
        self.z_ant = z_ant
        self.z_in_lna = z_in_lna
        self.antenna_gama = antenna_gama

    ### OPERATION

    def compute_for_freqs(self, a_freq_mhz):
        """!Compute transfer function for frequency a_freq_mhz

        @param a_freq_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        super().set_out_freq_mhz(a_freq_mhz)
        self._vswr.set_out_freq_mhz(a_freq_mhz)
        self._vswr.compute_s11()
        # update rho
        self._compute(self._vswr.s11)

    ### GETTER

    def get_fft_rho_3axis(self):
        """
        Get FFT of transfer function (TF) rho of LNA for 3 axis (port, TF)

        @return fft rho (port, TF): LNA transfer function
        """
        return self.rho123.T

    ### PLOT

    def plot_gama(self):  # pragma: no cover
        """
        plot of intermediate calculation gamma
        """
        plt.figure()
        plt.title("antenna_gama")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 0]), "k", label=r"0")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 1]), "y", label=r"1")
        plt.plot(self.freqs_out, np.abs(self.antenna_gama[:, 2]), "b", label=r"2")
        # plt.plot(self.freqs_out, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_z(self):  # pragma: no cover
        """
        plot of intermediate calculation z
        """
        plt.figure()
        plt.title("")
        plt.plot(self.freqs_out, np.abs(self.z_ant[:, 0]), label=r"$\mathregular{Z_{A}}$")
        plt.plot(self.freqs_out, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_lna(self):  # pragma: no cover
        """
        plot of FFT LNA transfer function rho=rho_1*rho_2*rho_3
        """
        l_col = ["k", "y", "b"]
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for port in range(3):
            plt.plot(self.freqs_in, self._dbs21_a[port], l_col[port])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.grid()
        plt.figure(figsize=(9, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for port in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho1[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.grid()
        plt.subplot(1, 3, 2)
        for port in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho2[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.grid()
        plt.subplot(1, 3, 3)
        for port in range(3):
            plt.plot(self.freqs_out, np.abs(self._rho3[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.grid()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    def plot_rho_kernel(self):
        """
        plot of LNA transfer function in time space
        """
        plt.figure()
        plt.title("Rho kernel")
        kernel_rho = sf.fftshift(sf.irfft(self.get_fft_rho_3axis()), axes=1)
        # kernel_rho = sf.irfft(self.get_fft_rho_3axis())
        print(kernel_rho.shape)
        # TODO: self.size_sig//2 or self.size_sig//2 -1 ?
        v_time = np.arange(self.size_sig, dtype=np.float64) - self.size_sig // 2
        dt_ns = 1e9 / (self.freqs_out[1] * self.size_sig * 1e6)
        dt_ns = 1
        v_time_ns = dt_ns * v_time
        plt.plot(v_time_ns, kernel_rho[0], "k", label="0")
        plt.plot(v_time_ns, kernel_rho[1], "y", label="1")
        plt.plot(v_time_ns, kernel_rho[2], "b", label="2")
        plt.xlabel("ns")
        plt.grid()
        plt.legend()


class VgaFilterBalunGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self):
        """ """
        super().__init__()
        self.data_filter = np.loadtxt(self._set_name_data_file())
        # Hz to MHz
        self.freqs_in = self.data_filter[:, 0] / 1e6
        self.dbs11 = 0
        self.dbs21 = 0
        self.dbs21_add_vga = 0

    ### INTERNAL

    def _set_name_data_file(self, axis=0):
        """!

        @param axis:
        """
        file_address = os.path.join("detector", "filterparameter", "1.s2p")
        return grand_add_path_data_model(file_address)

    ### GETTER

    def get_fft_vfb_3axis(self):
        """

        @return fft TF (port, TF): transfer function of filter, VGA and balum, same value on each axis
        """

        return np.array([self.fft_vgafilbal, self.fft_vgafilbal, self.fft_vgafilbal])

    ### OPERATION

    def compute_for_freqs(self, a_freq_mhz):
        """Compute transfer function for frequency a_freq_mhz

        @param a_freq_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(a_freq_mhz)
        # dB
        gain_vga = -1.5
        r_balun = 630 * 2 / 650
        freqs_in = self.freqs_in
        dbs11 = self.data_filter[:, 1]
        degs11 = self.data_filter[:, 2]
        mags11 = 10 ** (dbs11 / 20)
        res11 = mags11 * np.cos(np.deg2rad(degs11))
        ims11 = mags11 * np.sin(np.deg2rad(degs11))
        dbs21 = self.data_filter[:, 3]
        dbs21_add_vga = dbs21 + gain_vga + 20 * np.log10(r_balun)
        degs21 = self.data_filter[:, 4]
        mags21 = 10 ** (dbs21_add_vga / 20)
        res21 = mags21 * np.cos(np.deg2rad(degs21))
        ims21 = mags21 * np.sin(np.deg2rad(degs21))
        self.dbs11 = dbs11
        self.dbs21 = dbs21
        self.dbs21_add_vga = dbs21_add_vga
        # s11_complex
        s11_real = interpol_at_new_x(freqs_in, res11, self.freqs_out)
        s11_complex = s11_real + 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_out)
        # s21_complex
        s21_real = interpol_at_new_x(freqs_in, res21, self.freqs_out)
        s21_complex = s21_real + 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_out)
        # fft VGA filter balun
        self.fft_vgafilbal = (1 + s11_complex) * s21_complex

    ### PLOT

    def plot_filter(self):  # pragma: no cover
        """
        Do plot of intermediate value to define filter
        """
        plt.figure(figsize=(6, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        # S11 and S21 DB
        plt.subplot(1, 2, 1)
        plt.plot(self.freqs_in, self.dbs11)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(self.freqs_in, self.dbs21)
        plt.grid()
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of Filter", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # s21_add_vga
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.plot(self.freqs_in, self.dbs21_add_vga)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.title("S parameters of Filter add VGA", fontsize=15)
        plt.grid()
        plt.figure()
        plt.title("FFTS parameters of Filter add VGA", fontsize=15)
        plt.plot(self.freqs_out, np.abs(self.fft_vgafilbal))
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.grid()


class CableGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self):
        """ """
        super().__init__()
        self.data_cable = np.loadtxt(self._set_name_data_file())
        # Hz to MHz
        self.freqs_in = self.data_cable[:, 0] / 1e6
        self.dbs11 = 0
        self.dbs21 = 0

    ### INTERNAL

    def _set_name_data_file(self, axis=0):
        """!

        @param axis:
        """
        file_address = os.path.join("detector", "cableparameter", "cable.s2p")
        return grand_add_path_data_model(file_address)

    ### GETTER

    def get_fft_cable_3axis(self):
        """

        @return fft TF (port, TF): transfer function of cable, same value on each axis
        """
        return np.array([self.fft_cable, self.fft_cable, self.fft_cable])

    ### OPERATION

    def compute_for_freqs(self, a_freq_mhz):
        """Compute transfer function for frequency a_freq_mhz
        @param a_freq_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(a_freq_mhz)
        freqs_in = self.freqs_in
        dbs11 = self.data_cable[:, 1]
        degs11 = self.data_cable[:, 2]
        mags11 = 10 ** (dbs11 / 20)
        res11 = mags11 * np.cos(np.deg2rad(degs11))
        ims11 = mags11 * np.sin(np.deg2rad(degs11))
        dbs21 = self.data_cable[:, 3]
        degs21 = self.data_cable[:, 4]
        mags21 = 10 ** (dbs21 / 20)
        res21 = mags21 * np.cos(np.deg2rad(degs21))
        ims21 = mags21 * np.sin(np.deg2rad(degs21))
        self.dbs11 = dbs11
        self.dbs21 = dbs21
        # s11_complex
        s11_real = interpol_at_new_x(freqs_in, res11, self.freqs_out)
        s11_complex = s11_real + 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_out)
        # s21_complex
        s21_real = interpol_at_new_x(freqs_in, res21, self.freqs_out)
        s21_complex = s21_real + 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_out)
        self.fft_cable = (1 + s11_complex) * s21_complex

    ### PLOT

    def plot_cable(self):  # pragma: no cover
        """
        Do plot of intermediate value to define TF
        """
        plt.figure(figsize=(6, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 2, 1)
        plt.plot(self.freqs_in, self.dbs11)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.subplot(1, 2, 2)
        plt.plot(self.freqs_in, self.dbs21)
        plt.ylim(-10, 0)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of cable", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.figure()
        plt.title("FFT cable")
        plt.plot(self.freqs_out, np.abs(self.fft_cable))
        plt.xlabel("Frequency(MHz)", fontsize=15)


class RfChainGP300:
    """
    Facade for all elements in RF chain
    """

    def __init__(self):
        self.lna = LowNoiseAmplificatorGP300()
        self.vfb = VgaFilterBalunGP300()
        self.cable = CableGP300()
        self._total_tf = 0

    def compute_for_freqs(self, a_freq_mhz):
        """Compute transfer function for frequency a_freq_mhz

        @param a_freq_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.lna.compute_for_freqs(a_freq_mhz)
        self.vfb.compute_for_freqs(a_freq_mhz)
        self.cable.compute_for_freqs(a_freq_mhz)
        self._total_tf = (
            self.lna.get_fft_rho_3axis()
            * self.vfb.get_fft_vfb_3axis()
            * self.cable.get_fft_cable_3axis()
        )

    def get_tf_3axis(self):
        """Return transfer function for all elements in RF chain
        @return total TF (complex, (3,N)):
        """
        return self._total_tf

    def plot_kernel(self):  # pragma: no cover
        plt.figure()
        plt.title("Kernels associated to total transfer function of RF chain")
        kernel_0 = sf.fftshift(sf.irfft(self.get_tf_3axis()[0, :]))
        kernel_1 = sf.fftshift(sf.irfft(self.get_tf_3axis()[1, :]))
        kernel_2 = sf.fftshift(sf.irfft(self.get_tf_3axis()[2, :]))
        # kernel = sf.irfft(self.get_fft_rho_3axis())
        # TODO: self.size_sig//2 or self.size_sig//2 -1 ?
        v_time = np.arange(self.lna.size_sig, dtype=np.float64) - self.lna.size_sig // 2
        dt_ns = 1e9 / (self.lna.freqs_out[1] * self.lna.size_sig * 1e6)
        v_time_ns = dt_ns * v_time
        plt.plot(v_time_ns, kernel_0, "k", label="port 1")
        plt.plot(v_time_ns, kernel_1, "y", label="port 2")
        plt.plot(v_time_ns, kernel_2, "b", label="port 3")
        plt.xlabel("ns")
        plt.grid()
        plt.legend()

    def plot_tf(self):  # pragma: no cover
        freqs = self.lna.freqs_out
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xlim(40, 260)
        plt.title("Amplitude total transfer function")
        plt.plot(freqs, np.abs(self._total_tf[0]), "k", label="port 1")
        plt.plot(freqs, np.abs(self._total_tf[1]), "y", label="port 2")
        plt.plot(freqs, np.abs(self._total_tf[2]), "b", label="port 3")
        plt.xlabel("MHz")
        plt.grid()
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlim(40, 260)
        plt.title("Phase total transfer function")
        plt.plot(freqs, np.angle(self._total_tf[0], deg=True), "k", label="port 1")
        plt.plot(freqs, np.angle(self._total_tf[1], deg=True), "y", label="port 2")
        plt.plot(freqs, np.angle(self._total_tf[2], deg=True), "b", label="port 3")
        plt.xlabel("MHz")
        plt.ylabel("Deg")
        plt.grid()
        plt.legend()
