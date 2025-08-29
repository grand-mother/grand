"""
Created on 26/08/2025

Refactoring of module grand.sim.noise.galaxy by Colley Jean-Marc

Goal of refactoring:
* for each call of galactic_noise(), the function read files model on disk to do the same thing
* clearify FFT normalization
* clearify method of noise generation

So 
* Separate ASD computing (this module) and noise generation (galactic_ant_component.py) 

AND also
* Replace cubic interpolation by linear, more safe
* Simply check between what content of model galactic noise files and what we used finally 
* Add plot function in same module
* Create a file format for ASD
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data


def get_asd_galactic_ant_model(du_type="GP300"):
    """Return ASD of galactic signal through antenna of type "du_type", unit uV/sqrt(Hz).

    ..Authors:
      PengFei and Xidian group
      Modified by SN including different antenna models for leff

    :param du_type: Calculate the galactic noise for different antenna model simulations.
                 'GP300' (default) uses hfss simulations for leff
                 'GP300_nec' uses nec simulations for leff
                 'Gp300_mat' uses matlab simulations fro leff
    :return: ([MHz], [uV/sqrt(Hz)]) frequency, ASD of galactic noise through antenna of type "du_type"
    :rtype: float(nb_freq), float(nb_freq, axis=3, lst=24)
    """
    if du_type == "GP300":
        gala_file = grand_add_path_data("noise/PG_ALL_jifen.mat")
        Zant_file = grand_add_path_data("detector/RFchain_v2/Z_ant_3.2m.csv")
        gala_show = h5py.File(gala_file, "r")
        gala_power = np.array(gala_show["PG_ALL_jifen"])
        gala_power = np.transpose(gala_power, (2, 0, 1))  # Watt/Hz
        Poc2X = 1e6 * gala_power[:, :, 0]  # W
        Poc2Y = 1e6 * gala_power[:, :, 1]  # W
        Poc2Z = 1e6 * gala_power[:, :, 2]  # W

        zant = np.loadtxt(Zant_file, delimiter=",", skiprows=1)  # Skip header row if it exists
        # Extract real and imaginary parts and construct complex numbers
        zant_complex = np.column_stack(
            [
                zant[:, 1] + 1j * zant[:, 2],  # Z(1,1)
                zant[:, 3] + 1j * zant[:, 4],  # Z(2,2)
                zant[:, 5] + 1j * zant[:, 6],  # Z(3,3)
            ]
        )
        R = np.real(zant_complex)
        R_reshaped = R.T
        RantX = R_reshaped[0, :]
        RantY = R_reshaped[1, :]
        RantZ = R_reshaped[2, :]
        Voc2X = 4 * Poc2X * RantX[:, np.newaxis]
        Voc2Y = 4 * Poc2Y * RantY[:, np.newaxis]
        Voc2Z = 4 * Poc2Z * RantZ[:, np.newaxis]
        VocX = 1e6 * np.sqrt(Voc2X)  # in uV
        VocY = 1e6 * np.sqrt(Voc2Y)  # in uV
        VocZ = 1e6 * np.sqrt(Voc2Z)  # in uV
        gala_voltage = np.stack((VocX, VocY, VocZ), axis=1)
        # gala_psd_dbm = np.transpose(gala_show["psd_narrow_huatu"])
        # gala_power_dbm = np.transpose(
        #    gala_show["p_narrow_huatu"]
        # )  # SL, dbm per MHz, P=mean(V*V)/imp with imp=100 ohms
        # gala_voltage = np.transpose(
        #    gala_show["v_amplitude"]
        # )  # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V))
        ## gala_power_mag = np.transpose(gala_show["p_narrow"])
        gala_freq1 = np.arange(30.0, 251.0)
        gala_freq = gala_freq1.reshape(221, 1)

        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_freq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
    elif du_type == "GP300_nec":
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_nec.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_nec.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_nec.npy")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1))  # micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1)
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1))  # watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1))  # dBm per MHz
        gala_freq1 = np.arange(30.0, 251.0)
        gala_freq = gala_freq1.reshape(221, 1)
        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_frv_amplitude_infile = gala_voltage[:, :, lst]eq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
    elif du_type == "GP300_mat":
        print(du_type)
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_mat.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_mat.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_mat.npy")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1))  # micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1)
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1))  # watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1))  # dBm per MHz
        gala_freq1 = np.arange(30.0, 251.0)
        gala_freq = gala_freq1.reshape(221, 1)
        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_freq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""

    ##########################################################################
    # Here v_amplitude_infile is given in unit [uV/sqrt(Hz)] for all "du_type"
    ##########################################################################
    asd_ant_galactic = gala_voltage
    return gala_freq, asd_ant_galactic


def save_asd_galaxy(du_type, pf_name):
    fq, asd = get_asd_galactic_ant_model(du_type)
    # Create a NumPy structured array
    t_asd = np.dtype(
        {"names": ["fq", "asd"], "formats": ["f4", "(3,24)f4"], "titles": ["MHz", "uV/sqrt(Hz)"]}
    )
    sa_asd = np.zeros(len(fq), dtype=t_asd)
    sa_asd["fq"] = np.squeeze(fq)
    sa_asd["asd"] = asd
    np.save(pf_name, sa_asd)


def plot_check_psd_models(lst, axis):
    f_hfss, asd_hfss = get_asd_galactic_ant_model("GP300")
    print("HFSS shape :", asd_hfss.shape)
    f_nec, asd_nec = get_asd_galactic_ant_model("GP300_nec")
    print("NEC shape :", asd_nec.shape)
    f_matlab, asd_matlab = get_asd_galactic_ant_model("GP300_mat")
    print("Matlab shape :", asd_matlab.shape)

    plt.figure()
    plt.title(f"Model PSD galactic at LST {lst}, axis {axis}")
    plt.semilogy(f_hfss[1:-2], asd_hfss[1:-2, axis, lst] ** 2, label="Model HFSS")
    plt.semilogy(f_nec[1:-2], asd_nec[1:-2, axis, lst] ** 2, label="Model NEC")
    plt.semilogy(f_matlab[1:-2], asd_matlab[1:-2, axis, lst] ** 2, "*", label="Model Matlab")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
    plt.grid()
    plt.legend()


def plot_check_lst_sum_models(model="GP300"):
    _, asd = get_asd_galactic_ant_model(model)
    print("asd shape :", asd.shape)
    plt.figure()
    plt.title(f"Sum PSD ({model} model) for each axis")
    psd = asd[1:-1] ** 2
    psd_sum = psd.sum(axis=0)
    l_col = ["k", "y", "b"]
    lst = range(24)
    for i_a in range(3):
        plt.plot(lst, psd_sum[i_a], color=l_col[i_a], label=f"idx axis={i_a}")
        print(psd_sum[i_a])
    plt.plot(lst, psd.sum(axis=(0, 1)), "-*", label=f"Total all axis")
    print(psd.sum(axis=(0, 1)))
    plt.vlines(18, 200, 3200, label="idx 18")
    plt.ylabel("$\sum{PSD}$")
    plt.xlabel("index LST")
    plt.grid()
    plt.legend()


def plot_check_lst_models(lst, axis):
    f_hfss, asd_hfss = get_asd_galactic_ant_model("GP300")
    print("HFSS shape :", asd_hfss.shape)

    def sum_psd(axis, lst):
        return (asd_hfss[1:-2, axis, lst] ** 2).sum()

    plt.figure()
    plt.title(f"Model PSD galactic HFSS, axis {axis}")
    plt.semilogy(
        f_hfss[1:-2],
        asd_hfss[1:-2, axis, lst - 1] ** 2,
        color="k",
        label=f"LST idx={lst-1}, sum={sum_psd(axis, lst-1)}",
    )
    plt.semilogy(
        f_hfss[1:-2],
        asd_hfss[1:-2, axis, lst] ** 2,
        color="y",
        label=f"LST idx={lst}, sum={sum_psd(axis, lst)}",
    )
    plt.semilogy(
        f_hfss[1:-2],
        asd_hfss[1:-2, axis, lst + 1] ** 2,
        color="b",
        label=f"LST idx={lst+1}, sum={sum_psd(axis, lst+1)}",
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
    plt.grid()
    plt.legend()


if __name__ == "__main__":
    # plot_check_psd_models(18, 0)
    # plot_check_psd_models(18, 1)
    plot_check_psd_models(18, 2)
    # plot_check_lst_models(1,1)
    # plot_check_lst_models(17,0)
    # plot_check_lst_models(19,0)
    # plot_check_lst_models(17,1)
    # plot_check_lst_models(19,1)
    # plot_check_lst_models(17,2)
    # plot_check_lst_models(19,2)
    plot_check_lst_sum_models("GP300")
    save_asd_galaxy("GP300", "ASD_galaxy_ant_HFSS")

    plt.show()
