"""!
Simulation of the effects detector on the signal : like VSWR, LNA, cable, 

Reference document : "RF Chain simulation for GP300" 
                      by Xidian University, Pengfei Zhang and 
                      Pengxiong Ma, Chao Zhang, Rongjuan Wand, Xin Xu
"""

import os.path
import math
from logging import getLogger

from numpy.ma import log10, abs
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

from grand.num.signal import complex_expansion
from grand import grand_add_path_data_model

logger = getLogger(__name__)


def plot_csv_file(f_name, col_x, col_y, x_label="MHz", y_label="TBD"):
    plt.figure()
    plt.title(f"{f_name}, x:{col_x}, y:{col_y}")
    data = np.loadtxt(f_name)
    plt.plot(data[:, col_x], data[:, col_y])
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid()


def plot_csv_file_abs(f_name, col_x, col_y1, col_y2, x_label="MHz", y_label="TBD"):
    plt.figure()
    plt.title(f"{f_name}, x:{col_x}, ")
    data = np.loadtxt(f_name)
    plt.plot(data[:, col_x], np.abs(data[:, col_y1] + 1j * data[:, col_y2]))
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid()


class GenericProcessingDU:
    def __init__(self):
        """ """
        self.f_step = 1.0
        self.f_start = 30.0
        self.f_end = 250.0
        self.f_nb = 221

    ### SETTER

    def set_frequency_band(self, f_start, f_end, nb_band):
        """!

        @param f_start (float): [MHz]
        @param f_end (float): [MHz]
        @param nb_band (integer):
        """
        assert f_end > f_start > 0
        assert nb_band > 0
        self.f_start = f_start
        self.f_end = f_end
        self.f_step = (f_end - f_start) / (1.0 * nb_band)
        self.f_nb = nb_band + 1


class StandingWaveRatioGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaption Colley jm

    Class goals:
      * define VSWR value as s11 parameter
      * read only once data files
      * pre_compute interpolation
    """

    def __init__(self):
        super().__init__()
        self.s11 = np.zeros((self.f_nb, 3), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        lna_address = os.path.join("detector", "antennaVSWR", f"{axis+1}.s1p")
        return grand_add_path_data_model(lna_address)

    def compute_s11(self, unit=1):
        """!

        @param unit:
        """
        s11 = np.zeros((self.f_nb, 3), dtype=np.complex64)
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
            freqnew = np.linspace(self.f_start, self.f_end, self.f_nb)
            f_re = interpolate.interp1d(freq, re, kind="cubic")
            renew = f_re(freqnew)
            f_im = interpolate.interp1d(freq, im, kind="cubic")
            imnew = f_im(freqnew)
            s11[:, axis] = renew + 1j * imnew
        self.f_s11 = freqnew
        self.s11 = s11
        self.f_name = filename
        self.db_s11 = db_s11
        print(db_s11.shape, freq.shape)
        self.f_db_s11 = freq

    def plot_vswr(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"VSWR, s11 parameter")
        ax1.set_title("db")
        ax1.plot(self.f_db_s11, self.db_s11[0], label="0")
        ax1.plot(self.f_db_s11, self.db_s11[1], label="1")
        ax1.plot(self.f_db_s11, self.db_s11[2], label="2")
        ax1.set_ylabel(f"db")
        ax1.set_xlabel(f"[MHz]")
        ax1.grid()
        ax1.legend()
        ax2.set_title("abs(s_11)")
        ax2.plot(self.f_s11, np.abs(self.s11[:, 0]), label="0")
        ax2.plot(self.f_s11, np.abs(self.s11[:, 1]), label="1")
        ax2.plot(self.f_s11, np.abs(self.s11[:, 2]), label="2")
        ax2.set_ylabel(f"?")
        ax2.set_xlabel(f"[MHz]")
        ax2.grid()
        ax2.legend()


class LowNoiseAmplificatorGP300(GenericProcessingDU):
    """!
    @authors PengFei Zhang and Xidian group, GRANDLIB adaption Colley jm

    Class goals:
      * Perform the LNA filter on signal for each antenna of GP300
      * read only once LNA data files
      * pre_compute interpolation
    """

    def __init__(self, size_sig):
        """

        @param size_sig: size of the trace after
        """
        super().__init__()
        self.size_sig = size_sig
        self.data_lna = []
        for axis in range(3):
            lna = np.loadtxt(self._set_name_data_file(axis))
            # Hz to MHz
            lna[0] /= 1e6
            self.data_lna.append(lna)
        self.f_lna = lna[:, 0]
        self._pre_compute()

    ### INTERNAL

    def _set_name_data_file(self, axis):
        """!

        @param axis:
        """
        lna_address = os.path.join("detector", "LNASparameter", f"{axis+1}.s2p")
        return grand_add_path_data_model(lna_address)

    def _pre_compute(self, unit=1):
        """!
        compute what is possible without know response antenna

        @param unit: select LNA type stockage 0: [re, im],  1: [amp, arg]
        """
        lna_gama = np.zeros((self.size_sig, 3), dtype=np.complex64)
        lna_s21 = np.zeros((self.size_sig, 3), dtype=np.complex64)
        dbs21_a = np.zeros((3, self.f_lna.size))
        logger.debug(f"{dbs21_a.shape}")
        logger.debug(f"{self.data_lna[0].shape}")
        for axis in range(3):
            freq = self.data_lna[axis][:, 0]
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
                print(dbs21.shape)
                degs21 = self.data_lna[axis][:, 4]
                mags21 = 10 ** (dbs21 / 20)
                res21 = mags21 * np.cos(np.deg2rad(degs21))
                ims21 = mags21 * np.sin(np.deg2rad(degs21))
            dbs21_a[axis] = dbs21
            freqnew = np.arange(30, 251, 1)
            f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
            res11new = f_res11(freqnew)
            f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
            ims11new = f_ims11(freqnew)
            s11 = res11new + 1j * ims11new
            [f_expan, lna_gama[:, axis]] = complex_expansion(
                self.size_sig,
                self.f_step,
                self.f_start,
                self.f_end,
                s11,
            )
            f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
            res21new = f_res21(freqnew)
            f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
            ims21new = f_ims21(freqnew)
            s21 = res21new + 1j * ims21new
            [f_expan, lna_s21[:, axis]] = complex_expansion(
                self.size_sig,
                self.f_step,
                self.f_start,
                self.f_end,
                s21,
            )
        # Add attribut
        self.lna_gama = lna_gama
        self.lna_s21 = lna_s21
        self.dbs21_a = dbs21_a
        self.f_expan = f_expan

    ### OPERATION

    def update_with_s11(self, s11_short):
        """!
        @note:
          lll

        @param s11_short (N,3):
        """
        assert s11_short.shape[1] == 3
        z_0 = 50
        antenna_gama = np.zeros((self.size_sig, 3), dtype=np.complex64)
        for axis in range(3):
            [_, antenna_gama[:, axis]] = complex_expansion(
                self.size_sig,
                self.f_step,
                self.f_start,
                self.f_end,
                s11_short[:, axis],
            )
        z_ant = z_0 * (1 + antenna_gama) / (1 - antenna_gama)
        z_in_lna = z_0 * (1 + self.lna_gama) / (1 - self.lna_gama)
        self.rho1 = z_in_lna / (z_ant + z_in_lna)
        self.rho2 = (1 + self.lna_gama) / (1 - antenna_gama * self.lna_gama)
        self.rho3 = self.lna_s21 / (1 + self.lna_gama)
        self.z_ant = z_ant
        self.z_in_lna = z_in_lna
        self.antenna_gama = antenna_gama

    ### PLOT

    def plot_gama(self):
        plt.figure()
        plt.title("")
        plt.plot(self.f_expan, np.abs(self.antenna_gama[:, 0]), label=r"")
        # plt.plot(self.f_expan, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_z(self):
        plt.figure()
        plt.title("")
        plt.plot(self.f_expan, np.abs(self.z_ant[:, 0]), label=r"$\mathregular{Z_{A}}$")
        plt.plot(self.f_expan, np.abs(self.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid()
        plt.legend()

    def plot_lna(self):  # pragma: no cover
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(self.f_lna, self.dbs21_a[p])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.figure(figsize=(9, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for p in range(3):
            plt.plot(self.f_expan, abs(self.rho1[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.subplot(1, 3, 2)
        for p in range(3):
            plt.plot(self.f_expan, abs(self.rho2[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.subplot(1, 3, 3)
        for p in range(3):
            plt.plot(self.f_expan, abs(self.rho3[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)


def LNA_get(antennas11_complex_short, N, f0, unit, show_flag=False):
    """!

    @authors PengFei and Xidian group

    @param antennas11_complex_short
    @param N:
    @param f0:
    @param unit
    @param show_flag:
    """
    # = == == == == This program is used as a subroutine to complete the calculation and expansion of the LNA partial pressure coefficient == == == == =
    #  ----------------------input - ---------------------------------- %
    # antennas11_complex_short is the program after the interpolation of the antenna standing wave test result
    # LNA address, s2p file (delete the previous string of s2p file in advance, put the test results of the three ports in the LNASparameter folder, and name them 1 2 3 in turn, corresponding to xyz)
    # The unit of test frequency is hz
    # N is the extended length
    # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
    # f0 is the frequency resolution,
    # % ----------------------output - ---------------------------------- %
    # rho1 rho2 rho3
    def plot():  # pragma: no cover
        # drawing
        plt.figure()
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(freq, dbs21_a[p])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for p in range(3):
            plt.plot(f, abs(rho1[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.subplot(1, 3, 2)
        for p in range(3):
            plt.plot(f, abs(rho2[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.subplot(1, 3, 3)
        for p in range(3):
            plt.plot(f, abs(rho3[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    # characteristic impedance
    z0 = 50
    antenna_gama_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        # Antenna related parameter calculation
        antennas11_short = antennas11_complex_short
        f0 = 1
        f_start = 30
        f_end = 250
        [f, antenna_gama_complex[:, p]] = complex_expansion(
            N, f0, f_start, f_end, antennas11_short[:, p]
        )
    z_ant = z0 * (1 + antenna_gama_complex) / (1 - antenna_gama_complex)
    lna_gama_complex = np.zeros((N, 3), dtype=complex)  # 3 ports
    lna_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  LNA parameter
        str_p = str(p + 1)
        lna_address = os.path.join("detector", "LNASparameter", f"{ str_p}.s2p")
        lna_address = grand_add_path_data_model(lna_address)
        freq = np.loadtxt(lna_address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(lna_address, usecols=1)
            ims11 = np.loadtxt(lna_address, usecols=2)
            res21 = np.loadtxt(lna_address, usecols=3)
            ims21 = np.loadtxt(lna_address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))
        elif unit == 1:
            dbs11 = np.loadtxt(lna_address, usecols=1)
            degs11 = np.loadtxt(lna_address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)
            dbs21 = np.loadtxt(lna_address, usecols=3)
            degs21 = np.loadtxt(lna_address, usecols=4)
            mags21 = 10 ** (dbs21 / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)
        if p == 0:
            dbs21_a = np.zeros((3, len(freq)))
        dbs21_a[p] = dbs21
        # 30-250mh 1mhz
        freqnew = np.arange(30, 251, 1)
        f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
        res11new = f_res11(freqnew)
        f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
        ims11new = f_ims11(freqnew)
        s11_complex = res11new + 1j * ims11new
        [f, lna_gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)
        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, lna_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)
    z_in_lna = z0 * (1 + lna_gama_complex) / (1 - lna_gama_complex)
    # Partial pressure coefficient
    rho1 = z_in_lna / (z_ant + z_in_lna)
    rho2 = (1 + lna_gama_complex) / (1 - antenna_gama_complex * lna_gama_complex)
    rho3 = lna_s21_complex / (1 + lna_gama_complex)
    if show_flag:
        plot()
    return rho1, rho2, rho3


def filter_get(N, f0, unit, show_flag=False):
    """!

    @authors PengFei and Xidian group

    @param N:
    @param f0:
    @param unit:
    @param show_flag:
    """

    # = == == == == This program is used as a subroutine to complete the calculation and expansion of the S parameters of the cable and filter == == == == =
    #  ----------------------input - ---------------------------------- %
    # filter :path of s2p file
    # The filter test data is in the Filterparameter folder (delete the previous string of the s2p file in advance and name it 1, because the three port filters are the same)
    # The cable test data is in the cableparameter folder (delete the previous string of the s2p file in advance and name it cable because the three ports are the same)
    # The unit of test frequency is hz
    # N is the extended length
    # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
    # f0 is the frequency resolution,
    # % ----------------------output - ---------------------------------- %
    # cable_coefficient
    # filter_coefficient
    def plot():  # pragma: no cover
        plt.figure(figsize=(6, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 2, 1)
        for p in range(3):
            plt.plot(freq, dBs11_cable[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.subplot(1, 2, 2)
        for p in range(3):
            plt.plot(freq, dbs21_cable[p])
        plt.ylim(-10, 0)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of cable", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    def plot2():  # pragma: no cover
        plt.figure(figsize=(6, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 2, 1)
        for p in range(3):
            plt.plot(freq, dBs11[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.subplot(1, 2, 2)
        for p in range(3):
            plt.plot(freq, dbs21[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of Filter", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.figure()
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(freq, dbs21_add_vga[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.title("S parameters of Filter add VGA", fontsize=15)

    gain_vga = -1.5  # dB
    r_balun = 630 * 2 / 650
    # test filter without VGA
    # gain_vga = 0  # dB
    # r_balun = 1
    cable_gama_complex = np.zeros((N, 3), dtype=complex)  # 3 ports
    cable_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  cable参数
        # str_p = str(p + 1)
        # cable_address = ".//data//cableparameter//" + str_p + ".s2p"
        cable_address = os.path.join("detector", "cableparameter", "cable.s2p")
        cable_address = grand_add_path_data(cable_address)
        freq = np.loadtxt(cable_address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(cable_address, usecols=1)
            ims11 = np.loadtxt(cable_address, usecols=2)
            res21 = np.loadtxt(cable_address, usecols=3)
            ims21 = np.loadtxt(cable_address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))
            dbs11 = 20 * log10(abs(res11 + 1j * ims11))
        elif unit == 1:
            dbs11 = np.loadtxt(cable_address, usecols=1)
            degs11 = np.loadtxt(cable_address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)
            dbs21 = np.loadtxt(cable_address, usecols=3)
            degs21 = np.loadtxt(cable_address, usecols=4)
            mags21 = 10 ** (dbs21 / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)
        if p == 0:
            dbs21_cable = np.zeros((3, len(freq)))
            dBs11_cable = np.zeros((3, len(freq)))
        dbs21_cable[p] = dbs21
        dBs11_cable[p] = dbs11
        f_start = 30
        f_end = 250
        # Interpolation is a data of 30-250mhz interval 1mhz
        freqnew = np.arange(30, 251, 1)
        f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
        res11new = f_res11(freqnew)
        f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
        ims11new = f_ims11(freqnew)
        s11_complex = res11new + 1j * ims11new
        [f, cable_gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)
        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, cable_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)
    # drawing
    if show_flag:
        plot()
    cable_coefficient = (1 + cable_gama_complex) * cable_s21_complex
    #  =================filter=========================================
    filter_gama_complex = np.zeros((N, 3), dtype=complex)  # 3个端口
    filter_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  filter parameter
        # str_p = str(p + 1)
        # filter_address = ".//data//filterparameter//" + str_p + ".s2p"
        filter_address = os.path.join("detector", "filterparameter", "1.s2p")
        filter_address = grand_add_path_data(filter_address)
        freq = np.loadtxt(filter_address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(filter_address, usecols=1)
            ims11 = np.loadtxt(filter_address, usecols=2)
            res21_filter = np.loadtxt(filter_address, usecols=3)
            ims21_filter = np.loadtxt(filter_address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))
            dbs11 = 20 * log10(abs(res11 + 1j * ims11))
            dbs21_add_vga = dbs21 + gain_vga + 20 * log10(r_balun)
            mags21 = 10 ** (dbs21_add_vga / 20)
            degs21 = np.angle(res21_filter + ims21_filter)  # Phase radians
            res21 = mags21 * np.cos(degs21)
            ims21 = mags21 * np.sin(degs21)
        elif unit == 1:
            dbs11 = np.loadtxt(filter_address, usecols=1)
            degs11 = np.loadtxt(filter_address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)
            dbs21 = np.loadtxt(filter_address, usecols=3)
            dbs21_add_vga = dbs21 + gain_vga + 20 * log10(r_balun)
            degs21 = np.loadtxt(filter_address, usecols=4)
            mags21 = 10 ** (dbs21_add_vga / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)
        # Filter S parameter display
        if p == 0:
            dbs21_a = np.zeros((3, len(freq)))
            dBs11_a = np.zeros((3, len(freq)))
            dbs21_add_vga_a = np.zeros((3, len(freq)))
        dbs21_a[p] = dbs21
        dBs11_a[p] = dbs11
        dbs21_add_vga_a[p] = dbs21_add_vga
        # 插值为30-250mhz间隔1mhz一个数据
        freqnew = np.arange(30, 251, 1)
        f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
        res11new = f_res11(freqnew)
        f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
        ims11new = f_ims11(freqnew)
        s11_complex = res11new + 1j * ims11new
        [f, filter_gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)
        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, filter_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)
    if show_flag:
        plot2()
    filter_coefficient = (1 + filter_gama_complex) * filter_s21_complex
    return cable_coefficient, filter_coefficient
