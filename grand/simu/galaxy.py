"""!
Simulation of galaxy emission in radio frequency
"""

import random

import h5py
import numpy as np
import matplotlib.pyplot as plt

from grand.num.signal import complex_expansion, ifftget
from grand import grand_add_path_data


def galaxy_radio_signal(lst, size_out, freq_samp, freq_1, nb_ant, show_flag=False):
    """!
    This program is used as a subroutine to complete the calculation and
    expansion of galactic noise

    @authors PengFei and Xidian group

    @param lst：Select the galactic noise LST at the LST moment
    @param size_out (int): is the extended length
    @param freq_samp (float): is the frequency resolution,
    #TODO: freq_1 description not clear
    @param freq_1 (float): is the frequency point of the unilateral spectrum
    @param nb_ant (int): number of antennas
    @param show_flag (bool): print figure

    @return : v_complex_double, galactic_v_time
    """

    def plot():  # pragma: no cover
        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(gala_freq, gala_psd_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.subplot(1, 3, 3)
        for l_g in range(3):
            plt.plot(gala_freq, gala_voltage[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    gala_file = grand_add_path_data("sky/30_250galactic.mat")
    gala_show = h5py.File(gala_file, "r")
    gala_psd_dbm = np.transpose(gala_show["psd_narrow_huatu"])
    gala_power_dbm = np.transpose(gala_show["p_narrow_huatu"])
    gala_voltage = np.transpose(gala_show["v_amplitude"])
    # gala_power_mag = np.transpose(gala_show["p_narrow"])
    gala_freq = gala_show["freq_all"]
    if show_flag:
        plot()
    #
    f_start = 30
    f_end = 250
    # TODO: 221 is the number of frequency ? why ? and comment to explain
    nb_freq = 221
    v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
    galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
    galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
    galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
    v_amplitude = gala_voltage[:, :, lst - 1]
    a_nor = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    phase = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    v_complex = np.zeros((nb_ant, nb_freq, 3), dtype=complex)
    for l_ant in range(nb_ant):
        for l_fq in range(nb_freq):
            for l_axis in range(3):
                # Generates a normal distribution with 0 as the mean and
                # v_amplitude[l_fq, l_axis] as the standard deviation
                a_nor[l_ant, l_fq, l_axis] = np.random.normal(
                    loc=0, scale=v_amplitude[l_fq, l_axis]
                )
                # phase of random Gauss noise
                phase[l_ant, l_fq, l_axis] = 2 * np.pi * random.random()
                v_complex[l_ant, l_fq, l_axis] = abs(a_nor[l_ant, l_fq, l_axis] * size_out / 2)
                v_complex[l_ant, l_fq, l_axis] *= np.exp(1j * phase[l_ant, l_fq, l_axis])
    #
    for l_ant in range(nb_ant):
        for l_axis in range(3):
            [_, v_complex_double[l_ant, :, l_axis]] = complex_expansion(
                size_out,
                freq_samp,
                f_start,
                f_end,
                v_complex[l_ant, :, l_axis],
            )
        [galactic_v_time[l_ant], galactic_v_m_single[l_ant], galactic_v_p_single[l_ant]] = ifftget(
            v_complex_double[l_ant],
            size_out,
            freq_1,
            2,
        )
    return v_complex_double, galactic_v_time
