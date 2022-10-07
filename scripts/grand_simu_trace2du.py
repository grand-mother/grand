#! /usr/bin/env python3
"""!
Adaption of RF simulation chain for grandlib from
  * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py

@authors PengFei and Xidian group
"""

import os
import os.path as osp
import shutil
import math
import random
import argparse

import numpy
import numpy as np
from numpy.ma import log10, abs
from scipy import interpolate
import matplotlib.pyplot as plt

from grand.num.signal import fftget, ifftget
from grand.simu.du.elec_du import LNA_get, filter_get
from grand.simu.galaxy import galaxy_radio_signal
from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent
from grand.io.file_leff import TabulatedAntennaModel
from grand import grand_add_path_data, grand_get_path_root_pkg
from grand import ECEF, Geodetic, LTP, GRANDCS
import grand.manage_log as mlg
import grand.io.root_trees as groot

# showerdir = osp.join(grand_get_path_root_pkg(), "tests/simulation/data/zhaires")
# ShowerEvent.load(showerdir)
SHOWER = None
IDX_ANT = 0

path_ant = grand_add_path_data("model/detector/GP300Antenna_EWarm_leff.npy")
G_antenna_model_ew = TabulatedAntennaModel.load(path_ant)    
path_ant = grand_add_path_data("model/detector/GP300Antenna_SNarm_leff.npy")
G_antenna_model_sn = TabulatedAntennaModel.load(path_ant)    
path_ant = grand_add_path_data("model/detector/GP300Antenna_Zarm_leff.npy")
G_antenna_model_z = TabulatedAntennaModel.load(path_ant)    

# specific logger definition for script because __mane__ is "__main__"
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("info", log_file=None, log_stdout=True)



def halfcplx_fullcplx(v_half, even=True):
    '''!
    Return fft with full complex format where vector has half complex format,
    ie v_half=rfft(signal).
    
    @note:
      Numpy reference : https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html 
    
    @param v_half (array 1D complex): complex vector in half complex format, ie from rfft(signal)
    @param even (bool): True if size of signal is even
    
    @return (array 1D complex) : fft(signal) in full complex format
    '''
    if even:
        return np.concatenate((v_half, np.flip(np.conj(v_half[1:-1]))))
    return np.concatenate((v_half, np.flip(np.conj(v_half[1:]))))

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
    #N = math.ceil(fs)
    # JMC: remove fix length to 2000, use size of trace
    N = len(t)
    f0 = fs / N  # base frequency, Frequency resolution
    f = np.arange(0, N) * f0  # frequency sequence
    f1 = f[0: int(N / 2) + 1]
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
            t_cut[0: N - 500] = t[hang - (N - 500): hang]
            t_cut[N - 500: N] = t[hang: hang + 500]

            ex_cut[0: N - 500] = ex[hang - (N - 500): hang]
            ex_cut[N - 500: N] = ex[hang: hang + 500]

            ey_cut[0: N - 500] = ey[hang - (N - 500): hang]
            ey_cut[N - 500: N] = ey[hang: hang + 500]

            ez_cut[0: N - 500] = ez[hang - (N - 500): hang]
            ez_cut[N - 500: N] = ez[hang: hang + 500]
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
        
    logger.debug(f"ex_cut size is {len(ex_cut)}")
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

    return np.array(t_cut), np.array(ex_cut), np.array(ey_cut), np.array(ez_cut), fs, f0, f, f1, N


def dummy_CEL(idx_ant, e_theta, e_phi, N, f0, unit, show_flag=False):
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
# for i in range(3):  # Polarization i = 1, 2, 3 respectively represent xyz polarization
#         for p in range(3):
#             # Xyz polarization of a single port
#             Lce_complex_short[:, i, p] = e_radiation[:, p, i] / fenmu[:, p]
#             [f, Lce_complex_expansion[:, i, p]] = expan(N, f0, f1, f2, Lce_complex_short[:, i, p])    
        # for p in range(Lcelie):
        #     Voc_shower_complex[:, p] = Lce_complex[:, 0, p] * E_shower_fft[:, 0]  
    Lce_complex_expansion = np.zeros((N, 3, 3), dtype=complex)
    ants = get_antenna(idx_ant)
    for idx_ant in range(3):
        ants[idx_ant].effective_length(SHOWER.maximum, SHOWER.fields[idx_ant].electric, SHOWER.frame)
        for idx_axis in range(3):
            Lce_complex_expansion[:, idx_axis, idx_ant] = halfcplx_fullcplx(ants[idx_ant].dft_effv_len[idx_axis], (N%2)==0)
    
    # Interpolation: 30-250mhz interval 1mhz
    f_start = 30
    f_end = 250
    f_df = 1
    # +1 : f_end included
    f_size = (f_end - f_start) // f_df + 1
    s11_complex = np.zeros((f_size, 3), dtype=complex)
    for p in range(3):
        str_p = str(p + 1)
        # filename = ".//antennaVSWR//" + str_p + ".s1p"
        filename = grand_add_path_data(f"detector/antennaVSWR/{str_p}.s1p")
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
        if p == 0:
            dB = np.zeros((3, len(freq)))
        dB[p] = db
        #
        freqnew = np.arange(f_start, f_end + f_df, f_df)
        f_re = interpolate.interp1d(freq, re, kind="cubic")
        renew = f_re(freqnew)
        f_im = interpolate.interp1d(freq, im, kind="cubic")
        imnew = f_im(freqnew)
        s11_complex[:, p] = renew + 1j * imnew
    #
    return Lce_complex_expansion, s11_complex


def get_antenna(idx_ant):
    pos_ant = SHOWER.fields[idx_ant].electric.pos_xyz
    antenna_location = LTP(
        x=pos_ant.x,
        y=pos_ant.y,
        z=pos_ant.z,
        frame=SHOWER.frame,
    )
    antenna_frame = LTP(
        location=antenna_location,
        orientation="NWU",
        magnetic=True
    )
    ant_3d = [1, 2, 3]
    # EW
    ant_3d[0] = AntennaProcessing(G_antenna_model_ew, frame=antenna_frame)
    # SN
    ant_3d[1] = AntennaProcessing(G_antenna_model_sn, frame=antenna_frame)
    # Z
    ant_3d[2] = AntennaProcessing(G_antenna_model_z, frame=antenna_frame)
    return ant_3d


def main_xdu_rf(rootdir):
    """
    @param rootdir (path): directory with ZHAires simulation named Stshpxxxx
    """
    #TODO: this function is too long, split it     
    global SHOWER
    # =====================================================!!!Start the main program from here!!!=========================================
    os.chdir(rootdir)
    logger.info(rootdir)
    verse = []
    target = []
    logger.info(mlg.string_begin_script())
    for root in os.listdir(rootdir):
        verse.append(root)
            
    # =====================================If the folder starts with Stshp, add the target to be processed================================
    for item in verse:
        if item.startswith("Stshp_") == True:
            target.append(item)
       
    logger.info(f"target={target}")
    for i in range(0, len(target)):
        # file_dir = os.path.join("..", "data", target[i])
        file_dir = target[i]
        logger.debug(file_dir)
        logger.info(f'Read {file_dir}')
        SHOWER = ShowerEvent.load(file_dir)
        list1 = target[i].split("_")
        content = []
        target_trace = []
        primary = list1[3]
        energy = float(list1[4])
        e_theta = float(list1[5])
        e_phi = float(list1[6])
        case = float(list1[7])
        logger.info(f"primary is: {primary}")
        logger.info(f"energy is: {energy} Eev")
        logger.info(f"theta is: {e_theta} degree")
        logger.info(f"phi is: {e_phi} degree")
        logger.info(f"case num is: {case}")
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

            tvoltage = groot.VoltageEventTree() #Create tvoltage to be saved
            path_root_vfilter = outfile_vfilter + "/" + "a.root"

        # ==== Write particle type, energy level, angle, and event number into parameter.txt========================
        case_index = (
            "primary is:"
            +str(primary)
            +";           energy is: "
            +str(energy)
            +" Eev;          theta is: "
            +str(e_theta)
            +" degree;          phi is: "
            +str(e_phi)
            +" degree;          case num is:"
            +str(case)
        )
        path_par = os.path.join("result", target[i], "parameter.txt")
        with open(path_par, "w") as f:
            f.write(case_index)
            #TODO: JMC pb indentation ? 
            # <= ?
            source_root = os.path.join(file_dir, "antpos.dat")
            target_root = os.path.join("result", target[i])
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
                content.append(a)
            for b in content:
                # print(os.path.splitext(b)[1])
                if os.path.splitext(b)[1] == ".trace":
                    if b.startswith("a") == True:
                        target_trace.append(b)
            nb_ant = len(target_trace)
            #  ===========================start calculating===================
            [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(
                E_path, Ts, False
            )  # Signal interception
            #plt.show()
            Edata = ex_cut
            Edata = np.column_stack((Edata, ey_cut))
            Edata = np.column_stack((Edata, ez_cut))

            [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(
                Edata, N, f1, show_flag
            )  # Frequency domain signal

            # ======Galaxy noise power spectrum density, power, etc.=====================
            [galactic_v_complex_double, galactic_v_time] = galaxy_radio_signal(
                lst, N, f0, f1, nb_ant, show_flag
            )

            # =======================  cable  filter VGA balun=============================================
            [cable_coefficient, filter_coefficient] = filter_get(N, f0, 1, show_flag)
            # <= ?
        # ===============================Set global VoltageEvent params========================================================================
        if savetxt == 1:
            tvoltage.du_count = nb_ant
            tvoltage.run_number = case
            logger.info(f'{type(tvoltage.du_count)} {type(tvoltage.run_number)}')
            logger.info(f'ROOT IO: add run_number {tvoltage.run_number}')
        # ========================= Get first point and renormalisation of times =================================
        mintime = 1E100
        firstdet = 0
        for num in range(nb_ant):
            filename = os.path.join(file_dir, "a" + str(num) + ".trace")
            t = np.loadtxt(filename, usecols=0)
            tmin = t.min()
            if tmin <= mintime:
                mintime = tmin
                firstdet = num
        logger.info(f"ROOT IO: add first_du")
        tvoltage.first_du = firstdet
        tvoltage.time_seconds = 0
        tvoltage.time_nanoseconds = 0
        #tvoltage.time_seconds = numpy.uint32(int(mintime))
        #print(tvoltage.time_seconds)


        # ===============================Start loop calculation========================================================================
        for num in range(len(target_trace)):
            # air shower,input file
            E_path = os.path.join(file_dir, "a" + str(num) + ".trace")
            xunhuan = "a" + str(num) + ".trace"
            logger.info(f'Processing ============> {xunhuan}')
            # =======Equivalent length================
            [Lce_complex, antennas11_complex_short] = dummy_CEL(num, e_theta, e_phi, N, f0, 1, show_flag)
            Lcehang = Lce_complex.shape[0]
            Lcelie = Lce_complex.shape[2]
            logger.debug(Lce_complex.shape)
            # =================== LNA=====================================================
            [rou1_complex, rou2_complex, rou3_complex] = LNA_get(
                antennas11_complex_short, N, f0, 1, show_flag
            )            

            # Output file path and name
            #  ===========================start calculating===========================================================
            [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(
                E_path, Ts, show_flag
            )  # Signal interception
            Edata = ex_cut
            Edata = np.column_stack((Edata, ey_cut))
            Edata = np.column_stack((Edata, ez_cut))

            [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(
                Edata, N, f1, False
            )  # Frequency domain signal

            # =======Equivalent length================

            # ======Open circuit voltage of air shower=================
            Voc_shower_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
            # Frequency domain signal
            for p in range(Lcelie):
                Voc_shower_complex[:, p] = (
                    Lce_complex[:, 0, p] * E_shower_fft[:, 0]
                    +Lce_complex[:, 1, p] * E_shower_fft[:, 1]
                    +Lce_complex[:, 2, p] * E_shower_fft[:, 2]
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
                        Voc_shower_t[:, p] + galactic_v_time[random.randint(a=0, b=175),:, p]
                    )
                    Voc_noise_complex[:, p] = (
                        Voc_shower_complex[:, p]
                        +galactic_v_complex_double[random.randint(a=0, b=175),:, p]
                    )

            [Voc_noise_t_ifft, Voc_noise_m_single, Voc_noise_p_single] = ifftget(
                Voc_noise_complex, N, f1, 2
            )

            # ==================Voltage after LNA=====================================================
            V_LNA_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_LNA_complex[:, p] = rou1_complex[:, p] \
                                    * rou2_complex[:, p] \
                                    * rou3_complex[:, p] \
                                    * Voc_noise_complex[:, p]
                
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
                V_output1[:, 1:] = Voc_shower_t[:,:]
                np.savetxt(outfile_voc + "/"+xunhuan, V_output1, fmt="%.10e", delimiter=" ")
                # LNA
                V_output4 = np.zeros((N, Lcelie + 1))
                V_output4[:, 0] = t_cut
                V_output4[:, 1:] = V_LNA_t[:,:]
                np.savetxt(outfile_vlna + "/"+ xunhuan, V_output4, fmt="%.10e", delimiter=" ")
                # cable
                V_output2 = np.zeros((N, Lcelie + 1))
                V_output2[:, 0] = t_cut
                V_output2[:, 1:] = V_cable_t[:,:]
                np.savetxt(outfile_vcable+ "/" + xunhuan, V_output2, fmt="%.10e", delimiter=" ")
                # filter
                V_output3 = np.zeros((N, Lcelie + 1))
                V_output3[:, 0] = t_cut
                V_output3[:, 1:] = V_filter_t[:,:]
                np.savetxt(outfile_vfilter + "/"+ xunhuan, V_output3, fmt="%.10e", delimiter=" ")


            # ======================output to VoltageEvent root file=============================================
                trace_t = list(V_output3[:, 0])
                trace_x = list(V_output3[:, 1])
                trace_y = list(V_output3[:, 2])
                trace_z = list(V_output3[:, 3])
                logger.info(f"type trace {V_output3[:, 3].dtype} {V_output3[:, 3].shape}")
                # times in ns -> shifted to the first detection (as integer is needed we loose decimals !!!!)
                tvoltage.du_nanoseconds.append(round(trace_t[0]-mintime))
                tvoltage.du_seconds.append(0)
                # sampling in Mhz
                logger.info(f"ROOT IO: add du_id, trace_x ")
                sampling_ns = trace_t[1] - trace_t[0]
                sampling_mhz = int(1000/sampling_ns)
                tvoltage.adc_sampling_frequency.append(sampling_mhz)
                tvoltage.du_id.append(num)
                logger.info(f"type du_id {type(num)}")

                tvoltage.trace_x.append(trace_x)
                tvoltage.trace_y.append(trace_y)
                tvoltage.trace_z.append(trace_z)

            # ======================delete target_trace=============================================
        
        del target_trace
        plt.show()
    # ======================Save root file======================
    if savetxt == 1:
        ret = tvoltage.fill()
        logger.info(ret)
        ret = tvoltage.write(path_root_vfilter)
        logger.info(ret)
        logger.info(f"Wrote tvoltage in: {path_root_vfilter}")
    logger.info(mlg.string_end_script())

def test_get_antenna():
    ant = get_antenna()
    for key, val in ant.items():
        print(key, val.model.n_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ecpi script: run ecpi pipeline')
    parser.add_argument('path_simu',
                        help='path to ZHAires simulation')
    # retrieve argument
    args = parser.parse_args()    
    #main_xdu_rf("/home/jcolley/projet/grand_wk/binder/xdu")
    main_xdu_rf(args.path_simu)
    plt.show()
