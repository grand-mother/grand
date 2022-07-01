'''

'''

import os
import os.path
import shutil
import math
import random
import h5py

from numpy.ma import log10, abs
from scipy import interpolate
from scipy.fftpack import fft, ifft
import h5py
import numpy as np
import matplotlib.pyplot as plt

from grand.num.signal import complex_expansion, ifftget, fftget

# ===========================================LNAparameter get===========================================
def LNA_get(antennas11_complex_short, N, f0, unit, show_flag):
    # This Python file uses the following encoding: utf-8

    # from complex_expansion import expan

    # = == == == == This program is used as a subroutine to complete the calculation and expansion of the LNA partial pressure coefficient == == == == =
    #  ----------------------input - ---------------------------------- %
    # antennas11_complex_short is the program after the interpolation of the antenna standing wave test result
    # LNA address, s2p file (delete the previous string of s2p file in advance, put the test results of the three ports in the LNASparameter folder, and name them 1 2 3 in turn, corresponding to xyz)
    # The unit of test frequency is hz
    # N is the extended length
    # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
    # f0 is the frequency resolution,
    # % ----------------------output - ---------------------------------- %
    # rou1 rou2 rou3

    z0 = 50

    antenna_Gama_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        # Antenna related parameter calculation
        antennas11_short = antennas11_complex_short
        f0 = 1
        f_start = 30
        f_end = 250
        [f, antenna_Gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, antennas11_short[:, p])

    Zin_antenna = z0 * (1 + antenna_Gama_complex) / (1 - antenna_Gama_complex)

    LNA_Gama_complex = np.zeros((N, 3), dtype=complex)  # 3 ports
    LNA_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  LNA parameter
        str_p = str(p + 1)
        LNA_Address = os.path.join("LNASparameter", str_p + ".s2p")
        freq = np.loadtxt(LNA_Address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(LNA_Address, usecols=1)
            ims11 = np.loadtxt(LNA_Address, usecols=2)
            res21 = np.loadtxt(LNA_Address, usecols=3)
            ims21 = np.loadtxt(LNA_Address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))

        elif unit == 1:
            dbs11 = np.loadtxt(LNA_Address, usecols=1)
            degs11 = np.loadtxt(LNA_Address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)

            dbs21 = np.loadtxt(LNA_Address, usecols=3)
            degs21 = np.loadtxt(LNA_Address, usecols=4)
            mags21 = 10 ** (dbs21 / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)

        if p == 0:
            dBs21 = np.zeros((3, len(freq)))
        dBs21[p] = dbs21

        # 插值为30-250mhz间隔1mhz一个数据
        freqnew = np.arange(30, 251, 1)
        f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
        res11new = f_res11(freqnew)
        f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
        ims11new = f_ims11(freqnew)
        s11_complex = res11new + 1j * ims11new
        [f, LNA_Gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, LNA_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)

    Zin_LNA = z0 * (1 + LNA_Gama_complex) / (1 - LNA_Gama_complex)

    # Partial pressure coefficient
    rou1 = Zin_LNA / (Zin_antenna + Zin_LNA)
    rou2 = (1 + LNA_Gama_complex) / (1 - antenna_Gama_complex * LNA_Gama_complex)
    rou3 = LNA_s21_complex / (1 + LNA_Gama_complex)

    if show_flag == 1:
        # drawing
        plt.figure()
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(freq, dBs21[p])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.show()

        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for p in range(3):
            plt.plot(f, abs(rou1[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.subplot(1, 3, 2)
        for p in range(3):
            plt.plot(f, abs(rou2[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.subplot(1, 3, 3)
        for p in range(3):
            plt.plot(f, abs(rou3[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return rou1, rou2, rou3


# ===============================================Filterparameter get====================================
def filter_get(N, f0, unit, show_flag):
    # This Python file uses the following encoding: utf-8

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

    Gain_VGA = -1.5  # dB
    r_balun = 630 * 2 / 650
    # test filter without VGA
    # Gain_VGA = 0  # dB
    # r_balun = 1
    cable_Gama_complex = np.zeros((N, 3), dtype=complex)  # 3 ports
    cable_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  cable参数
        # str_p = str(p + 1)
        # cable_Address = ".//data//cableparameter//" + str_p + ".s2p"
        cable_Address = os.path.join("cableparameter", "cable.s2p")
        freq = np.loadtxt(cable_Address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(cable_Address, usecols=1)
            ims11 = np.loadtxt(cable_Address, usecols=2)
            res21 = np.loadtxt(cable_Address, usecols=3)
            ims21 = np.loadtxt(cable_Address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))
            dbs11 = 20 * log10(abs(res11 + 1j * ims11))

        elif unit == 1:
            dbs11 = np.loadtxt(cable_Address, usecols=1)
            degs11 = np.loadtxt(cable_Address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)

            dbs21 = np.loadtxt(cable_Address, usecols=3)
            degs21 = np.loadtxt(cable_Address, usecols=4)
            mags21 = 10 ** (dbs21 / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)

        if p == 0:
            dBs21_cable = np.zeros((3, len(freq)))
            dBs11_cable = np.zeros((3, len(freq)))
        dBs21_cable[p] = dbs21
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
        [f, cable_Gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, cable_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)
    # drawing
    if show_flag == 1:
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
            plt.plot(freq, dBs21_cable[p])
        plt.ylim(-10, 0)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of cable", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    cable_coefficient = (1 + cable_Gama_complex) * cable_s21_complex

    #  =================filter=========================================
    filter_Gama_complex = np.zeros((N, 3), dtype=complex)  # 3个端口
    filter_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  filter parameter
        # str_p = str(p + 1)
        # filter_Address = ".//data//filterparameter//" + str_p + ".s2p"
        filter_Address = os.path.join("filterparameter", "1.s2p")
        freq = np.loadtxt(filter_Address, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            res11 = np.loadtxt(filter_Address, usecols=1)
            ims11 = np.loadtxt(filter_Address, usecols=2)
            res21_filter = np.loadtxt(filter_Address, usecols=3)
            ims21_filter = np.loadtxt(filter_Address, usecols=4)
            dbs21 = 20 * log10(abs(res21 + 1j * ims21))
            dbs11 = 20 * log10(abs(res11 + 1j * ims11))
            dbs21_add_VGA = dbs21 + Gain_VGA + 20 * log10(r_balun)
            mags21 = 10 ** (dbs21_add_VGA / 20)
            degs21 = np.angle(res21_filter + ims21_filter)  # Phase radians
            res21 = mags21 * np.cos(degs21)
            ims21 = mags21 * np.sin(degs21)
        elif unit == 1:
            dbs11 = np.loadtxt(filter_Address, usecols=1)
            degs11 = np.loadtxt(filter_Address, usecols=2)
            mags11 = 10 ** (dbs11 / 20)
            res11 = mags11 * np.cos(degs11 / 180 * math.pi)
            ims11 = mags11 * np.sin(degs11 / 180 * math.pi)

            dbs21 = np.loadtxt(filter_Address, usecols=3)
            dbs21_add_VGA = dbs21 + Gain_VGA + 20 * log10(r_balun)
            degs21 = np.loadtxt(filter_Address, usecols=4)
            mags21 = 10 ** (dbs21_add_VGA / 20)
            res21 = mags21 * np.cos(degs21 / 180 * math.pi)
            ims21 = mags21 * np.sin(degs21 / 180 * math.pi)

        # Filter S parameter display
        if p == 0:
            dBs21 = np.zeros((3, len(freq)))
            dBs11 = np.zeros((3, len(freq)))
            dBs21_add_VGA = np.zeros((3, len(freq)))
        dBs21[p] = dbs21
        dBs11[p] = dbs11
        dBs21_add_VGA[p] = dbs21_add_VGA
        # 插值为30-250mhz间隔1mhz一个数据
        freqnew = np.arange(30, 251, 1)
        f_res11 = interpolate.interp1d(freq, res11, kind="cubic")
        res11new = f_res11(freqnew)
        f_ims11 = interpolate.interp1d(freq, ims11, kind="cubic")
        ims11new = f_ims11(freqnew)
        s11_complex = res11new + 1j * ims11new
        [f, filter_Gama_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, filter_s21_complex[:, p]] = complex_expansion(N, f0, f_start, f_end, s21_complex)
    # drawing
    if show_flag == 1:
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
            plt.plot(freq, dBs21[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of Filter", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        plt.figure()
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for p in range(3):
            plt.plot(freq, dBs21_add_VGA[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.title("S parameters of Filter add VGA", fontsize=15)
        plt.show()

    filter_coefficient = (1 + filter_Gama_complex) * filter_s21_complex

    return cable_coefficient, filter_coefficient
