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

from grand.num.signal import complex_expansion, ifftget

# =========================================galacticnoise get=============================================
def galaxy_radio_sig(lst, N, f0, f1, show_flag):
    # This Python file uses the following encoding: utf-8

    # = == == == == This program is used as a subroutine to complete the calculation and expansion of galactic noise == == == == =
    #  ----------------------input - ---------------------------------- %
    # lst：Select the galactic noise LST at the LST moment
    # N is the extended length
    # f0 is the frequency resolution, f1 is the frequency point of the unilateral spectrum
    # % ----------------------output - ---------------------------------- %
    # v_complex_double, galactic_v_time

    GALAshowFile = "30_250galactic.mat"
    GALAshow = h5py.File(GALAshowFile, "r")
    GALApsd_dbm = np.transpose(GALAshow["psd_narrow_huatu"])
    GALApower_dbm = np.transpose(GALAshow["p_narrow_huatu"])
    GALAvoltage = np.transpose(GALAshow["v_amplitude"])
    GALApower_mag = np.transpose(GALAshow["p_narrow"])
    GALAfreq = GALAshow["freq_all"]

    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for g in range(3):
            plt.plot(GALAfreq, GALApsd_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.subplot(1, 3, 2)
        for g in range(3):
            plt.plot(GALAfreq, GALApower_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.subplot(1, 3, 3)
        for g in range(3):
            plt.plot(GALAfreq, GALAvoltage[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    f_start = 30
    f_end = 250

    R = 50
    v_complex_double = np.zeros((176, N, 3), dtype=complex)
    galactic_v_time = np.zeros((176, N, 3), dtype=float)
    galactic_v_m_single = np.zeros((176, int(N / 2) + 1, 3), dtype=float)
    galactic_v_p_single = np.zeros((176, int(N / 2) + 1, 3), dtype=float)
    unit_uv = 1e6
    V_amplitude = 2 * np.sqrt(GALApower_mag * R) * unit_uv

    aa = np.zeros((176, 221, 3), dtype=float)
    phase = np.zeros((176, 221, 3), dtype=float)
    v_complex = np.zeros((176, 221, 3), dtype=complex)
    for mm in range(176):
        for ff in range(221):
            for pp in range(3):
                aa[mm, ff, pp] = np.random.normal(loc=0, scale=V_amplitude[ff, pp])
                phase[mm, ff, pp] = 2 * np.pi * random.random()  # 加入一随机高斯白噪声的相位
                v_complex[mm, ff, pp] = abs(aa[mm, ff, pp] * N / 2) * np.exp(1j * phase[mm, ff, pp])

    pass

    for kk in range(176):

        for port in range(3):
            [f, v_complex_double[kk, :, port]] = complex_expansion(
                N, f0, f_start, f_end, v_complex[kk, :, port]
            )
            # print(v_complex_double[k, :, port])
        [galactic_v_time[kk], galactic_v_m_single[kk], galactic_v_p_single[kk]] = ifftget(
            v_complex_double[kk], N, f1, 2
        )

        return v_complex_double, galactic_v_time
