"""!
Simulation of galaxy emission in radio frequency
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt

from grand.num.signal import interpol_at_new_x
from grand import grand_add_path_data


def galaxy_radio_signal(f_lst, size_out, freqs_mhz, nb_ant, show_flag=False):
    """!
    This program is used as a subroutine to complete the calculation and
    expansion of galactic noise

    @authors PengFei and Xidian group

    :param lst：Select the galactic noise LST at the LST moment
    :param size_out (int): is the extended length
    :param freq_samp (float): is the frequency resolution,
    #TODO: freq_1 description not clear
    :param freq_1 (float): is the frequency point of the unilateral spectrum
    :param nb_ant (int): number of antennas
    :param show_flag (bool): print figure

    @return : v_complex_double, galactic_v_time
    """
    # TODO: why lst is an integer ?
    lst = int(f_lst)

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
            plt.plot(
                gala_freq, 1e6 * np.sqrt(2 * 100 * pow(10, gala_power_dbm[:, l_g, lst] / 10) * 1e-3)
            )  # SL
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

    gala_file = grand_add_path_data("model/sky/30_250galactic.mat")
    gala_show = h5py.File(gala_file, "r")
    gala_psd_dbm = np.transpose(gala_show["psd_narrow_huatu"])
    gala_power_dbm = np.transpose(
        gala_show["p_narrow_huatu"]
    )  # SL, dbm per MHz, P=mean(V*V)/imp with imp=100 ohms
    gala_voltage = np.transpose(
        gala_show["v_amplitude"]
    )  # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V))
    # gala_power_mag = np.transpose(gala_show["p_narrow"])
    gala_freq = gala_show["freq_all"]
    if show_flag:
        plot()
        plt.show()

    """f_start = 30
    f_end = 250
    # TODO: 221 is the number of frequency ? why ? and comment to explain
    nb_freq = 221
    v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
    galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
    galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
    galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
    v_amplitude_infile = gala_voltage[:, :, lst - 1]

    # SL
    nb_freq = len(freqs_mhz)
    freq_res = freqs_mhz[1] - freqs_mhz[0]
    v_amplitude_infile = v_amplitude_infile * freq_res
    v_amplitude = np.zeros((nb_freq, 3))
    v_amplitude[:, 0] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 0], freqs_mhz)
    v_amplitude[:, 1] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 1], freqs_mhz)
    v_amplitude[:, 2] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 2], freqs_mhz)

    a_nor = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    phase = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    v_complex = np.zeros((nb_ant, 3, nb_freq), dtype=complex)
    for l_ant in range(nb_ant):
        for l_fq in range(nb_freq):
            for l_axis in range(3):
                # Generates a normal distribution with 0 as the mean and
                # v_amplitude[l_fq, l_axis] as the standard deviation
                a_nor[l_ant, l_fq, l_axis] = np.random.normal(
                    loc=0, scale=v_amplitude[l_fq, l_axis]
                )
                # phase of random Gauss noise
                phase[l_ant, l_fq, l_axis] = 2 * np.pi * np.random.random_sample()
                # SL *size_out is because default scipy fft is normalised backward, *1/2 is because mean(cos(x)*cos(x)))
                v_complex[l_ant, l_axis, l_fq] = abs(a_nor[l_ant, l_fq, l_axis] * size_out / 2)
                v_complex[l_ant, l_axis, l_fq] *= np.exp(1j * phase[l_ant, l_fq, l_axis])

    return v_complex
