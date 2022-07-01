"""
Simulation of galaxy emission in radio frequency
"""

import random

import h5py
import numpy as np
import matplotlib.pyplot as plt

from grand.num.signal import complex_expansion, ifftget


def galaxy_radio_signal(lst, size_out, f0, f1, show_flag=False):
    """!
    This program is used as a subroutine to complete the calculation and
    expansion of galactic noise

    @authors PengFei and Xidian group

    @param lst：Select the galactic noise LST at the LST moment
    @param size_out (int): is the extended length
    @param f0 (float): is the frequency resolution,
    @param f1 (float): is the frequency point of the unilateral spectrum
    @param show_flag (bool): print figure

    @return : v_complex_double, galactic_v_time
    """
    GALAshowFile = "30_250galactic.mat"
    GALAshow = h5py.File(GALAshowFile, "r")
    GALApsd_dbm = np.transpose(GALAshow["psd_narrow_huatu"])
    GALApower_dbm = np.transpose(GALAshow["p_narrow_huatu"])
    GALAvoltage = np.transpose(GALAshow["v_amplitude"])
    GALApower_mag = np.transpose(GALAshow["p_narrow"])
    GALAfreq = GALAshow["freq_all"]
    if show_flag:
        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(GALAfreq, GALApsd_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(GALAfreq, GALApower_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.subplot(1, 3, 3)
        for l_g in range(3):
            plt.plot(GALAfreq, GALAvoltage[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    #
    f_start = 30
    f_end = 250
    R = 50
    v_complex_double = np.zeros((176, size_out, 3), dtype=complex)
    galactic_v_time = np.zeros((176, size_out, 3), dtype=float)
    galactic_v_m_single = np.zeros((176, int(size_out / 2) + 1, 3), dtype=float)
    galactic_v_p_single = np.zeros((176, int(size_out / 2) + 1, 3), dtype=float)
    unit_uv = 1e6
    V_amplitude = 2 * np.sqrt(GALApower_mag * R) * unit_uv
    aa = np.zeros((176, 221, 3), dtype=float)
    phase = np.zeros((176, 221, 3), dtype=float)
    v_complex = np.zeros((176, 221, 3), dtype=complex)
    for mm in range(176):
        for ff in range(221):
            for pp in range(3):
                aa[mm, ff, pp] = np.random.normal(loc=0, scale=V_amplitude[ff, pp])
                phase[mm, ff, pp] = 2 * np.pi * random.random()
                v_complex[mm, ff, pp] = abs(aa[mm, ff, pp] * size_out / 2)
                v_complex[mm, ff, pp] *= np.exp(1j * phase[mm, ff, pp])
    #
    for kk in range(176):
        for port in range(3):
            [f, v_complex_double[kk, :, port]] = complex_expansion(
                size_out,
                f0,
                f_start,
                f_end,
                v_complex[kk, :, port],
            )
        [galactic_v_time[kk], galactic_v_m_single[kk], galactic_v_p_single[kk]] = ifftget(
            v_complex_double[kk],
            size_out,
            f1,
            2,
        )
    return v_complex_double, galactic_v_time
