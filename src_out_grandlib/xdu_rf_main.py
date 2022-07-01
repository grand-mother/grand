"""!
Adaption of RF simulation chain for grandlib from
  * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py

@authors PengFei and Xidian group
"""

import os
import os.path
import shutil
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data
from grand.num.signal import fftget, ifftget
from grand.simu.elec_du import LNA_get, filter_get
from grand.simu.galaxy import galaxy_radio_signal

# ==============================time domain shower Edata get===============================
def time_data_get(filename, Ts, show_flag):
    # This Python file uses the following encoding: utf-8

    # This program, as a subroutine, completes the reading of the shower time domain signal, ===========================================================
    # intercepts part of the signal or extends the length of the signal, and generates parameters according to the time domain requirements == == == == =
    #  ----------------------input - ---------------------------------- %
    # filename:path
    # Ts:time interval
    # % show_flag:flag of showing picture

    # % ----------------------output - ---------------------------------- %
    # % t:time sequence, unit:ns
    # % E_shower_cut:Corresponding to the triple polarization component of the time series，unit:uv
    # % fs % sampling frequency, unit:MHZ
    # % f0 % base frequency, Frequency resolution，unit:MHZ
    # % f  % frequency sequence，unit:MHz
    # % f1 % Unilateral spectrum frequency sequence，unit:MHz

    t = np.loadtxt(filename, usecols=(0))
    ex = np.loadtxt(filename, usecols=(1))
    ey = np.loadtxt(filename, usecols=(2))
    ez = np.loadtxt(filename, usecols=(3))

    # = == == == == == == == == == =Time-frequency parameter generation == == == == == == == == == == == == == == == == == == == == ==
    # = == == =In order to make the frequency resolution 1MHz, the number of sampling points = sampling frequency == == == =
    fs = 1 / Ts * 1000  # sampling frequency, MHZ
    N = math.ceil(fs)
    f0 = fs / N  # base frequency, Frequency resolution
    f = np.arange(0, N) * f0  # frequency sequence
    f1 = f[0 : int(N / 2) + 1]
    # Take only half, pay attention to odd and even numbers, odd numbers: the floor(N / 2 + 1) is conjugated to floor(N / 2 + 1) + 1;
    # Even number: floor(N / 2 + 1)-1 and floor(N / 2 + 1) + 1 are conjugated;

    # = == == == Change the original signal length to be the same as N======================
    t_cut = np.zeros((N))
    ex_cut = np.zeros((N))
    ey_cut = np.zeros((N))
    ez_cut = np.zeros((N))

    lt = len(t)
    if N <= lt:
        # ============================In order to avoid not getting the peak value, judge whether the peak value is within N == == == == == == == =
        posx = np.argmax(ex)
        posy = np.argmax(ey)
        posz = np.argmax(ez)
        hang = max(posx, posy, posz)
        if hang >= N:
            t_cut[0 : N - 500] = t[hang - (N - 500) : hang]
            t_cut[N - 500 : N] = t[hang : hang + 500]

            ex_cut[0 : N - 500] = ex[hang - (N - 500) : hang]
            ex_cut[N - 500 : N] = ex[hang : hang + 500]

            ey_cut[0 : N - 500] = ey[hang - (N - 500) : hang]
            ey_cut[N - 500 : N] = ey[hang : hang + 500]

            ez_cut[0 : N - 500] = ez[hang - (N - 500) : hang]
            ez_cut[N - 500 : N] = ez[hang : hang + 500]
        else:
            t_cut[0:N] = t[0:N]
            ex_cut[0:N] = ex[0:N]
            ey_cut[0:N] = ey[0:N]
            ez_cut[0:N] = ez[0:N]
    else:
        t_cut[0:lt] = t[0:]
        ex_cut[0:lt] = ex[0:]
        ey_cut[0:lt] = ey[0:]
        ez_cut[0:lt] = ez[0:]
        a = t[-1] + Ts
        b = t[-1] + Ts * (
            N - lt + 2
        )  # There is always a problem with the accuracy of decimal addition and subtraction. After +2, no matter if the decimal becomes .9999 or .000001, it is guaranteed to get the first n-lt number.
        add = np.arange(a, b, Ts)
        t_cut[lt:] = add[: (N - lt)]
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        plt.plot(t_cut, ex_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ex(uv/m)", fontsize=15)

        plt.subplot(1, 3, 2)
        plt.plot(t_cut, ey_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ey(uv/m)", fontsize=15)

        plt.subplot(1, 3, 3)
        plt.plot(t_cut, ez_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ez(uv/m)", fontsize=15)

        plt.suptitle("E Fields of Shower in Time Domain", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return np.array(t_cut), np.array(ex_cut), np.array(ey_cut), np.array(ez_cut), fs, f0, f, f1, N



def dummy_CEL(e_theta, e_phi, N, f0, unit, show_flag=False):
    # This Python file uses the following encoding: utf-8

    # from complex_expansion import expan

    # = == == == == This program is used as a subroutine to complete the calculation and expansion of the 30-250MHz complex equivalent length == == == == =
    #  ----------------------input- ---------------------------------- %
    # filename address, S1P file (delete the previous string of s1p file in advance, put the test results of the three ports in the antennaVSWR folder, and name them 1 2 3 in turn)
    # % show_flag :flag of showing picture
    # N is the extended length
    # e_theta, e_phi is the direction of incidence
    # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
    # f0 is the frequency resolution,
    # % ----------------------output - ---------------------------------- %
    # f frequency sequence, the default unit is MHz
    # Lce_complex_expansion is the equivalent length of a specific incident direction
    # s11_complex is the antenna test data
    
    return toto, tutu


def main_xdu_rf(rootdir):

    # =====================================================!!!Start the main program from here!!!=========================================
    os.chdir(rootdir)
    print(rootdir)
    verse = []
    target = []

    for root in os.listdir(rootdir):
        # print(root)
        # print(os.path.isdir(root))
        # if os.path.isdir(root) == True:            #Determine whether it is a folder
        verse.append(root)
        print(root)
    print("verse", verse)
    print(type(verse))
    # =====================================If the folder starts with Stshp, add the target to be processed================================
    for item in verse:
        if item.startswith("Stshp_") == True:
            target.append(item)
        print(item)
    print("target", target)

    for i in range(0, len(target)):
        # file_dir = os.path.join("..", "data", target[i])
        file_dir = target[i]
        print(file_dir)
        list1 = target[i].split("_")
        content = []
        target_trace = []

        primary = list1[3]
        energy = float(list1[4])
        e_theta = float(list1[5])
        e_phi = float(list1[6])
        case = float(list1[7])

        print("primary is:", primary)
        print("energy is:", energy, "Eev")
        print("theta is:", e_theta, "degree")
        print("phi is:", e_phi, "degree")
        print("case num is:", case)
        # ==================================switch===========================================

        savetxt = 1  # savetxt=0 Don't save output file ,savetxt=1 save output file
        show_flag = 0
        noise_flag = 0

        if savetxt == 1:
            if noise_flag == 0:
                trunc = os.path.join("result", target[i], "output_withoutnoise")
                outfile_voc = os.path.join(trunc, "voc")
                outfile_vlna = os.path.join(trunc, "Vlna")
                outfile_vcable = os.path.join(trunc, "Vlna_cable")
                outfile_vfilter = os.path.join(trunc, "Vfilter")
            elif noise_flag == 1:
                trunc = os.path.join("result", target[i], "output")
                outfile_voc = os.path.join(trunc, "voc")
                outfile_vlna = os.path.join(trunc, "Vlna")
                outfile_vcable = os.path.join(trunc, "Vlna_cable")
                outfile_vfilter = os.path.join(trunc, "Vfilter")
            os.makedirs(outfile_voc, exist_ok=True)
            os.makedirs(outfile_voc, exist_ok=True)
            os.makedirs(outfile_vlna, exist_ok=True)
            os.makedirs(outfile_vcable, exist_ok=True)
            os.makedirs(outfile_vfilter, exist_ok=True)
        # ==== Write particle type, energy level, angle, and event number into parameter.txt========================
        case_index = (
            "primary is:"
            + str(primary)
            + ";           energy is: "
            + str(energy)
            + " Eev;          theta is: "
            + str(e_theta)
            + " degree;          phi is: "
            + str(e_phi)
            + " degree;          case num is:"
            + str(case)
        )
        path_par = os.path.join("result", target[i], "parameter.txt")
        with open(path_par, "w") as f:
            f.write(case_index)

            source_root = os.path.join(file_dir, "antpos.dat")
            target_root = os.path.join("result", target[i])
            print("before shutil.copy")
            print(source_root, target_root)
            shutil.copy(source_root, target_root)

            # ===================================Arbitrary input file first generates input parameters needed by subroutine================================================
            #  ================================Change according to actual situation==========================================
            # Select the galactic noise LST at the LST moment
            # lst = int(input("please input lst:"))
            # demo = int(input("please input demo number:"))
            lst = 18
            Ts = 0.5  # Manually enter the same time interval as the .trace file
            randomn = 0
            E_path = os.path.join(file_dir, "a" + str(randomn) + ".trace")

            for a in os.listdir(file_dir):
                # print(root)
                # print(os.path.isdir(root))
                # if os.path.isdir(root) == True:            #判断是否为文件夹
                content.append(a)
            for b in content:
                # print(os.path.splitext(b)[1])
                if os.path.splitext(b)[1] == ".trace":
                    if b.startswith("a") == True:
                        target_trace.append(b)

            #  ===========================start calculating===================
            [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(
                E_path, Ts, show_flag
            )  # Signal interception

            Edata = ex_cut
            Edata = np.column_stack((Edata, ey_cut))
            Edata = np.column_stack((Edata, ez_cut))

            [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(
                Edata, N, f1, show_flag
            )  # Frequency domain signal
            # =======Equivalent length================
            [Lce_complex, antennas11_complex_short] = dummy_CEL(e_theta, e_phi, N, f0, 1, show_flag)

            Lcehang = Lce_complex.shape[0]
            Lcelie = Lce_complex.shape[2]
            # ======Galaxy noise power spectrum density, power, etc.=====================
            [galactic_v_complex_double, galactic_v_time] = galaxy_radio_signal(
                lst, N, f0, f1, show_flag
            )
            # =================== LNA=====================================================
            [rou1_complex, rou2_complex, rou3_complex] = LNA_get(
                antennas11_complex_short, N, f0, 1, show_flag
            )
            # =======================  cable  filter VGA balun=============================================
            [cable_coefficient, filter_coefficient] = filter_get(N, f0, 1, show_flag)

        # ===============================Start loop calculation========================================================================
        for num in range(len(target_trace)):
            # air shower,input file
            E_path = os.path.join(file_dir, "a" + str(num) + ".trace")
            xunhuan = "a" + str(num) + "_trace.txt"

            # Output file path and name
            #  ===========================start calculating===========================================================
            [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(
                E_path, Ts, show_flag
            )  # Signal interception

            Edata = ex_cut
            Edata = np.column_stack((Edata, ey_cut))
            Edata = np.column_stack((Edata, ez_cut))

            [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(
                Edata, N, f1, show_flag
            )  # Frequency domain signal

            # =======Equivalent length================

            # ======Open circuit voltage of air shower=================
            Voc_shower_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
            # Frequency domain signal
            for p in range(Lcelie):
                Voc_shower_complex[:, p] = (
                    Lce_complex[:, 0, p] * E_shower_fft[:, 0]
                    + Lce_complex[:, 1, p] * E_shower_fft[:, 1]
                    + Lce_complex[:, 2, p] * E_shower_fft[:, 2]
                    + 0
                )
            # time domain signal
            [Voc_shower_t, Voc_shower_m_single, Voc_shower_p_single] = ifftget(
                Voc_shower_complex, N, f1, 2
            )

            # ======Galaxy noise power spectrum density, power, etc.=====================

            # ===========Voltage with added noise=======================================
            Voc_noise_t = np.zeros((Lcehang, Lcelie))
            Voc_noise_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
            for p in range(Lcelie):
                if noise_flag == 0:
                    Voc_noise_t[:, p] = Voc_shower_t[:, p]
                    Voc_noise_complex[:, p] = Voc_shower_complex[:, p]
                elif noise_flag == 1:
                    Voc_noise_t[:, p] = (
                        Voc_shower_t[:, p] + galactic_v_time[random.randint(a=0, b=175), :, p]
                    )
                    Voc_noise_complex[:, p] = (
                        Voc_shower_complex[:, p]
                        + galactic_v_complex_double[random.randint(a=0, b=175), :, p]
                    )

            [Voc_noise_t_ifft, Voc_noise_m_single, Voc_noise_p_single] = ifftget(
                Voc_noise_complex, N, f1, 2
            )

            # ==================Voltage after LNA=====================================================
            V_LNA_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_LNA_complex[:, p] = (
                    rou1_complex[:, p]
                    * rou2_complex[:, p]
                    * rou3_complex[:, p]
                    * Voc_noise_complex[:, p]
                    + 0
                )
            [V_LNA_t, V_LNA_m_single, V_LNA_p_single] = ifftget(V_LNA_complex, N, f1, 2)

            # ======================Voltage after  cable=============================================
            V_cable_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_cable_complex[:, p] = V_LNA_complex[:, p] * cable_coefficient[:, p] + 0
            [V_cable_t, V_cable_m_single, V_cable_p_single] = ifftget(V_cable_complex, N, f1, 2)

            # ======================Voltage after filter=============================================
            V_filter_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_filter_complex[:, p] = (
                    V_LNA_complex[:, p] * cable_coefficient[:, p] * filter_coefficient[:, p] + 0
                )
            [V_filter_t, V_filter_m_single, V_filter_p_single] = ifftget(V_filter_complex, N, f1, 2)
            # ====================Voltage after ADC======================================
            Length_AD = 14  # Significant bit of the value, plus a sign bit in addition to this
            Range_AD = 1.8 * 1e6  # Vpp,unit:uv
            delta = Range_AD / 2 / (2 ** (Length_AD - 1))
            V_ADC_t = np.sign(V_filter_t) * np.floor(abs(V_filter_t) / delta) * delta
            # ======================save .txt=============================================
            if savetxt == 1:
                # time--ns,Voltage----uV
                # Open circuit voltage
                V_output1 = np.zeros((N, Lcelie + 1))
                V_output1[:, 0] = t_cut
                V_output1[:, 1:] = Voc_shower_t[:, :]
                np.savetxt(outfile_voc + xunhuan, V_output1, fmt="%.10e", delimiter=" ")
                # LNA
                V_output4 = np.zeros((N, Lcelie + 1))
                V_output4[:, 0] = t_cut
                V_output4[:, 1:] = V_LNA_t[:, :]
                np.savetxt(outfile_vlna + xunhuan, V_output4, fmt="%.10e", delimiter=" ")
                # cable
                V_output2 = np.zeros((N, Lcelie + 1))
                V_output2[:, 0] = t_cut
                V_output2[:, 1:] = V_cable_t[:, :]
                np.savetxt(outfile_vcable + xunhuan, V_output2, fmt="%.10e", delimiter=" ")
                # filter
                V_output3 = np.zeros((N, Lcelie + 1))
                V_output3[:, 0] = t_cut
                V_output3[:, 1:] = V_filter_t[:, :]
                np.savetxt(outfile_vfilter + xunhuan, V_output3, fmt="%.10e", delimiter=" ")

            # ======================delete target_trace=============================================

        del target_trace


if __name__ == "__main__":
    print("XDU")
    main_xdu_rf("/home/jcolley/projet/grand_wk/binder/xdu")
