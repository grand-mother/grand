# From : https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py
# Authprs : PengFei
# This Python file uses the following encoding: utf-8
import h5py
import os
import os.path
import shutil
from scipy.fftpack import fft
from scipy.fftpack import ifft
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.ma import log10, abs
from scipy import interpolate
import random

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
    f1 = f[0:int(N / 2) + 1]
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
            ex_cut[0: N] = ex[0: N]
            ey_cut[0: N] = ey[0: N]
            ez_cut[0: N] = ez[0: N]
    else:
        t_cut[0:lt] = t[0:]
        ex_cut[0: lt] = ex[0:]
        ey_cut[0: lt] = ey[0:]
        ez_cut[0: lt] = ez[0:]
        a = t[-1] + Ts
        b = t[-1] + Ts * (N - lt + 2)  # There is always a problem with the accuracy of decimal addition and subtraction. After +2, no matter if the decimal becomes .9999 or .000001, it is guaranteed to get the first n-lt number.
        add = np.arange(a, b, Ts)
        t_cut[lt:] = add[:(N - lt)]
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
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


# ================================================FFT get=============================================
def fftget(data_ori, N, f1, show_flag):
    # This Python file uses the following encoding: utf-8

    # = == == == == This program is used as a subroutine to complete the FFT of data and generate parameters according to requirements == == == == =
    #  ----------------------input- ---------------------------------- %
    # % data_ori:time domain data, matrix form
    # % show_flag:flag of showing picture
    # % N:number of FFT points
    # % f1:Unilateral frequency
    # % ----------------------output - ---------------------------------- %
    # % data_fft:Frequency domain complex data
    # % data_fft_m_single:Frequency domain amplitude unilateral spectrum
    # % data_fft:Frequency domain phase

    lienum = data_ori.shape[1]
    data_fft = np.zeros((N, lienum), dtype=complex)
    data_fft_m = np.zeros((int(N), lienum))
    # data_fft_m_single = np.zeros((int(N/2), lienum))
    # data_fft_p = np.zeros((int(N), lienum))
    # data_fft_p_single = np.zeros((int(N/2), lienum))

    for i in range(lienum):
        data_fft[:, i] = fft(data_ori[:, i])

        data_fft_m[:, i] = abs(data_fft[:, i]) * 2 / N  # Amplitude
        data_fft_m[0] = data_fft_m[0] / 2

        data_fft_m_single = data_fft_m[0: len(f1)]  # unilateral

        data_fft_p = np.angle(data_fft, deg=True)  # phase
        data_fft_p = np.mod(data_fft_p, 2 * 180)
        # data_fft_p_deg = np.rad2deg(data_fft_p)
        data_fft_p_single = data_fft_p[0: len(f1)]

    string = np.array(['x', 'y', 'z'])
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        for j in range(lienum):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, j + 1)
            plt.plot(f1, data_fft_m_single[:, j])
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("E" + string[j] + "(uv/m)", fontsize=15)
            plt.suptitle("Electric Field Spectrum", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # The smaller the value, the farther away
        plt.show()

        plt.figure(figsize=(9, 3))
        for j in range(lienum):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, j + 1)
            plt.plot(f1, data_fft_p_single[:, j])
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("E" + string[j] + "(deg)", fontsize=15)
            plt.suptitle("Electric Field Phase", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return np.array(data_fft), np.array(data_fft_m_single), np.array(data_fft_p_single)


# =====================================IFFT get=================================================
def ifftget(data_ori, N, f1, true):
    # This Python file uses the following encoding: utf-8


    # %= == == == == This program is used as a subroutine to complete the Fourier change of data and generate parameters according to requirements == == == == =
    # % ----------------------input - ---------------------------------- %
    # % data_ori:Frequency domain data, complex numbers
    # % true  1 indicates that the complex number is synthesized, that is, the amplitude is the real amplitude. 2 indicates that the complex number is obtained after Fourier transform;
    # % N:number of FFT points
    # % t:time sequence
    # ns
    # % ----------------------output - ---------------------------------- %
    # % data_ifft :time domain data

    lienum = data_ori.shape[1]

    # %= == == == == == == == == == == == == == == First draw the spectrum phase == == == == == ==
    data_ori_m = np.zeros((int(N), lienum))
    data_ori_p = np.zeros((int(N), lienum))
    if true == 1:
        for i in range(lienum):
            data_ori_m[:, i] = abs(data_ori[:, i])  # Amplitude
            data_ori_m_single = data_ori_m[0: len(f1)]  # unilateral

            data_ori_p[:, i] = np.angle(data_ori[:, i], deg=True)  # phase
            data_ori_p[:, i] = np.mod(data_ori_p[:, i], 2 * 180)
            data_ori_p_single = data_ori_p[0: len(f1)]

    elif true == 2:
        for i in range(lienum):
            data_ori_m[:, i] = abs(data_ori[:, i]) * 2 / N
            data_ori_m[0] = data_ori_m[0] / 2

            data_ori_m_single = data_ori_m[0: len(f1)]  # 单边

            data_ori_p = np.angle(data_ori, deg=True)  # 相位
            data_ori_p = np.mod(data_ori_p, 2 * 180)  # -pi到pi转为0到2pi
            data_ori_p_single = data_ori_p[0: len(f1)]

    # % % 时域
    data_ifft = np.zeros((N, lienum))
    for i in range(lienum):
        data_ifft[:, i] = ifft(data_ori[:, i]).real

    return np.array(data_ifft), np.array(data_ori_m_single), np.array(data_ori_p_single)
#==================================interpolation=======================================
def inter(data_complex_five,e_theta,e_phi):
    # This Python file uses the following encoding: utf-8

    # =================This subroutine is an interpolation procedure for a five-dimensional function in a specific theta phi direction======================
    # data_complex_five is the original data, 5 dimensions
    # e_theta,e_phi is the incident direction

    #    Four adjacent points
    down_theta = math.floor(e_theta)
    up_theta = math.ceil(e_theta)
    down_phi = math.floor(e_phi)
    up_phi = math.ceil(e_phi)

    a = abs(round(e_theta) - e_theta)
    b = abs(round(e_phi) - e_phi)

    numf=data_complex_five.shape[0]
    #    interpolation
    data_complex = np.zeros((181, 361), dtype=complex)
    data_new = np.zeros((numf, 3, 3), dtype=complex)
    for i in range(numf):
        for j in range(3):
            for k in range(3):
                data_complex[:, :] = data_complex_five[i, j, k, :, :]
                L1 = data_complex[down_theta, down_phi]
                L2 = data_complex[up_theta, down_phi]
                L3 = data_complex[down_theta, up_phi]
                L4 = data_complex[up_theta, up_phi]
                rt1 = (e_theta - down_theta) / 1.
                rt0 = 1.0 - rt1
                rp1 = (e_phi - down_phi) / 1.
                rp0 = 1.0 - rp1

                data_new[i, j, k] = rt0 * rp0 * L1 + rt1 * rp0 * L2 + rt0 * rp1 * L3 + rt1 * rp1 * L4
    return np.array(data_new)
# ==========================================complex expansion========================
def expan(N, f0, f1, f2, data):
    # This Python file uses the following encoding: utf-8

    # = == == == == This procedure is used as a subroutine to complete the expansion of the spectrum == == == == =
    # % N is the number of frequency points, that is, the spectrum that needs to be expanded
    # % f0 is the frequency step, MHz
    # % f1 is the starting frequency of the spectrum to be expanded, f2 is the cutoff frequency of the spectrum to be expanded
    # % The program only considers that the length of the expanded data is less than floor(N / 2), such as N = 10, the length of the expanded data <= 5; N = 9, the length of the expanded data <= 4
    # data 1 dimension

    f = np.arange(0, N) * f0  # Frequency sequence
    effective = len(data)
    delta_start = abs(f - f1)  # Difference from f1
    delta_end = abs(f - f2)  # Difference with f2
    f_hang_start = np.where(delta_start == min(delta_start))  # The row with the smallest difference
    f_hang_start = f_hang_start[0][0]
    f_hang_end = np.where(delta_end == min(delta_end))
    f_hang_end = f_hang_end[0][0]
    data_expansion = np.zeros((N), dtype=complex)
    if f_hang_start == 0:
        data_expansion[0] = data[0]
        add = np.arange(f_hang_end + 1, N - effective + 1, 1)
        duichen = np.arange(N - 1, N - effective + 1 - 1, -1)
        data_expansion[add] = 0
        data_expansion[f_hang_start: f_hang_end + 1] = data
        data_expansion[duichen] = data[1:].conjugate()
    else:
        a1 = np.arange(0, f_hang_start - 1 + 1, 1).tolist()
        a2 = np.arange(f_hang_end + 1, N - f_hang_start - effective + 1, 1).tolist()
        a3 = np.arange(N - f_hang_start + 1, N, 1).tolist()  # Need to make up 0;
        add = a1 + a2 + a3
        add = np.array(add)
        duichen = np.arange(N - f_hang_start, N - f_hang_start - effective, -1)
        data_expansion[add] = 0
        data_expansion[f_hang_start: f_hang_end + 1] = data[:]
        data_expansion[duichen] = data.conjugate()

    return f, data_expansion

# ==================================equivalent=========================================
def CEL(e_theta, e_phi, N, f0, unit, show_flag):
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

    # Complex electric field 30-250MHz
    REfile = ".//Complex_RE.mat"
    RE = h5py.File(REfile, 'r')
    RE_zb = np.transpose(RE['data_rE_ALL'])
    re_complex = RE_zb.view('complex')
    f_radiation = np.transpose(RE['f_radiation'])  # mhz
    effective = max(f_radiation.shape[0], f_radiation.shape[1])

    e_radiation = inter(re_complex, e_theta, e_phi)

    # 测试s1p
    s11_complex = np.zeros((effective, 3), dtype=complex)  # 3 ports
    for p in range(3):
        str_p = str(p + 1)
        filename = ".//antennaVSWR//" + str_p + ".s1p"
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

        # Interpolation is a data of 30-250mhz interval 1mhz
        freqnew = np.arange(30, 251, 1)
        f_re = interpolate.interp1d(freq, re, kind="cubic")
        renew = f_re(freqnew)
        f_im = interpolate.interp1d(freq, im, kind="cubic")
        imnew = f_im(freqnew)
        s11_complex[:, p] = renew + 1j * imnew

    # %Reduced current
    z0 = 50
    a1 = 1
    I_complex = 1 / math.sqrt(z0) * (1 - s11_complex) * a1

    # %Denominator
    eta = 120 * math.pi
    c = 3 * 1e8
    f_unit = 1e6
    lamda = c / (f_radiation * f_unit)  # m
    lamda = np.transpose(lamda)
    k = 2 * math.pi / lamda
    fenmu = 1j * (I_complex / (2 * lamda) * eta)

    # Equivalent length
    # Extend the frequency band
    f1 = f_radiation[0][0]
    f2 = f_radiation[0][-1]

    Lce_complex_short = np.zeros((effective, 3, 3), dtype=complex)
    Lce_complex_expansion = np.zeros((N, 3, 3), dtype=complex)
    for i in range(3):  # Polarization i = 1, 2, 3 respectively represent xyz polarization
        for p in range(3):
            # Xyz polarization of a single port
            Lce_complex_short[:, i, p] = e_radiation[:, p, i] / fenmu[:, p]
            [f, Lce_complex_expansion[:, i, p]] = expan(N, f0, f1, f2, Lce_complex_short[:, i, p])
    if show_flag == 1:
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        for p in range(3):
            plt.plot(freq, dB[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right', fontsize=15)
        plt.title(r"$\mathregular{S_{11}}$" + " of Antenna", fontsize=15)
        plt.show()

        string = np.array(['X', 'Y', 'Z'])
        plt.figure(figsize=(9, 3))
        for i in range(3):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, i + 1)
            for j in range(3):
                plt.plot(f, abs(Lce_complex_expansion[:, j, i]))
            plt.legend(["X polarization", "Y polarization", "Z polarization"], loc='upper right', fontsize=9)
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("Equivalent length / m", fontsize=15)
            plt.title(r"$\mathregular{L_e}$" + " in " + string[i] + " direction", fontsize=15)
            plt.xlim(30, 250)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # The smaller the value, the farther away
        plt.show()
    return Lce_complex_expansion, s11_complex


# =========================================galacticnoise get=============================================
def gala(lst, N, f0, f1, show_flag):
    # This Python file uses the following encoding: utf-8


    # = == == == == This program is used as a subroutine to complete the calculation and expansion of galactic noise == == == == =
    #  ----------------------input - ---------------------------------- %
    # lst：Select the galactic noise LST at the LST moment
    # N is the extended length
    # f0 is the frequency resolution, f1 is the frequency point of the unilateral spectrum
    # % ----------------------output - ---------------------------------- %
    # v_complex_double, galactic_v_time

    GALAshowFile = ".//30_250galactic.mat"
    GALAshow = h5py.File(GALAshowFile, 'r')
    GALApsd_dbm = np.transpose(GALAshow['psd_narrow_huatu'])
    GALApower_dbm = np.transpose(GALAshow['p_narrow_huatu'])
    GALAvoltage = np.transpose(GALAshow['v_amplitude'])
    GALApower_mag = np.transpose(GALAshow['p_narrow'])
    GALAfreq = GALAshow['freq_all']

    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 3, 1)
        for g in range(3):
            plt.plot(GALAfreq, GALApsd_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.subplot(1, 3, 2)
        for g in range(3):
            plt.plot(GALAfreq, GALApower_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.subplot(1, 3, 3)
        for g in range(3):
            plt.plot(GALAfreq, GALAvoltage[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
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
            [f, v_complex_double[kk, :, port]] = expan(N, f0, f_start, f_end, v_complex[kk, :, port])
            # print(v_complex_double[k, :, port])
        [galactic_v_time[kk], galactic_v_m_single[kk], galactic_v_p_single[kk]] = ifftget(v_complex_double[kk], N, f1, 2)

        return v_complex_double, galactic_v_time


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
        [f, antenna_Gama_complex[:, p]] = expan(N, f0, f_start, f_end, antennas11_short[:, p])

    Zin_antenna = z0 * (1 + antenna_Gama_complex) / (1 - antenna_Gama_complex)

    LNA_Gama_complex = np.zeros((N, 3), dtype=complex)  # 3 ports
    LNA_s21_complex = np.zeros((N, 3), dtype=complex)
    for p in range(3):
        #  LNA parameter
        str_p = str(p + 1)
        LNA_Address = ".//LNASparameter//" + str_p + ".s2p"
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
        [f, LNA_Gama_complex[:, p]] = expan(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, LNA_s21_complex[:, p]] = expan(N, f0, f_start, f_end, s21_complex)

    Zin_LNA = z0 * (1 + LNA_Gama_complex) / (1 - LNA_Gama_complex)

    # Partial pressure coefficient
    rou1 = Zin_LNA / (Zin_antenna + Zin_LNA)
    rou2 = (1 + LNA_Gama_complex) / (1 - antenna_Gama_complex * LNA_Gama_complex)
    rou3 = LNA_s21_complex / (1 + LNA_Gama_complex)

    if show_flag == 1:
        # drawing
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        for p in range(3):
            plt.plot(freq, dBs21[p])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right', fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.show()

        plt.figure(figsize=(9, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 3, 1)
        for p in range(3):
            plt.plot(f, abs(rou1[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.subplot(1, 3, 2)
        for p in range(3):
            plt.plot(f, abs(rou2[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.subplot(1, 3, 3)
        for p in range(3):
            plt.plot(f, abs(rou3[:, p]))
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
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
        cable_Address = ".//cableparameter//cable.s2p"
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
        [f, cable_Gama_complex[:, p]] = expan(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, cable_s21_complex[:, p]] = expan(N, f0, f_start, f_end, s21_complex)
    # drawing
    if show_flag == 1:
        plt.figure(figsize=(6, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 2, 1)
        for p in range(3):
            plt.plot(freq, dBs11_cable[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right')
        plt.subplot(1, 2, 2)
        for p in range(3):
            plt.plot(freq, dBs21_cable[p])
        plt.ylim(-10, 0)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right')
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
        filter_Address = ".//filterparameter//1.s2p"
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
        [f, filter_Gama_complex[:, p]] = expan(N, f0, f_start, f_end, s11_complex)

        f_res21 = interpolate.interp1d(freq, res21, kind="cubic")
        res21new = f_res21(freqnew)
        f_ims21 = interpolate.interp1d(freq, ims21, kind="cubic")
        ims21new = f_ims21(freqnew)
        s21_complex = res21new + 1j * ims21new
        [f, filter_s21_complex[:, p]] = expan(N, f0, f_start, f_end, s21_complex)
    # drawing
    if show_flag == 1:
        plt.figure(figsize=(6, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 2, 1)
        for p in range(3):
            plt.plot(freq, dBs11[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right')
        plt.subplot(1, 2, 2)
        for p in range(3):
            plt.plot(freq, dBs21[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right')
        plt.suptitle("S parameters of Filter", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        for p in range(3):
            plt.plot(freq, dBs21_add_VGA[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right')
        plt.title("S parameters of Filter add VGA", fontsize=15)
        plt.show()

    filter_coefficient = (1 + filter_Gama_complex) * filter_s21_complex

    return cable_coefficient, filter_coefficient


# ===================================make_new=========================================
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # Determine whether there is a folder, if it does not exist, create a folder
        os.makedirs(path)  # makedirs If the path does not exist when creating the file, it will be created
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("--- folder already exist!  ---")






#=====================================================!!!Start the main program from here!!!=========================================
rootdir = "..//data//"
print(rootdir)
verse = []
target = []
for root in os.listdir(rootdir):
        # print(root)
    # print(os.path.isdir(root))
    # if os.path.isdir(root) == True:            #Determine whether it is a folder
        verse.append(root)
        # print(root)
# print(verse)
# print(type(verse))
#=====================================If the folder starts with Stshp, add the target to be processed================================
for item in verse:
    if item.startswith('Stshp_') == True:
        target.append(item)
# print(target)

for i in range(0, len(target)):
    file_dir = "..//data//" + target[i]
    print(file_dir)
    list1 = target[i].split('_')
    content = []
    target_trace = []

    primary = list1[3]
    energy = float(list1[4])
    e_theta = float(list1[5])
    e_phi = float(list1[6])
    case = float(list1[7])


    print("primary is:" , primary)
    print("energy is:", energy, "Eev")
    print("theta is:", e_theta, "degree")
    print("phi is:", e_phi, "degree")
    print("case num is:", case)
#==================================switch===========================================

    savetxt = 1  # savetxt=0 Don't save output file ,savetxt=1 save output file
    show_flag = 0
    noise_flag = 0


    if savetxt == 1:
        if noise_flag == 0:
            outfile_voc =  "..//result//" + target[i] + '//output_withoutnoise//voc'
            outfile_vlna =  "..//result//" + target[i] + '//output_withoutnoise//Vlna'
            outfile_vcable =  "..//result//" + target[i] + '//output_withoutnoise//Vlna_cable'
            outfile_vfilter =  "..//result//" + target[i] + '//output_withoutnoise//Vfilter'
        elif noise_flag == 1:
            outfile_voc = "..//result//" + target[i] + '//output//voc'
            outfile_vlna = "..//result//" + target[i] + '//output//Vlna'
            outfile_vcable = "..//result//" + target[i] + '//output//Vlna_cable'
            outfile_vfilter = "..//result//" + target[i] + '//output//Vfilter'

        mkdir(outfile_voc)
        mkdir(outfile_vlna)
        mkdir(outfile_vcable)
        mkdir(outfile_vfilter)
#==========================Write particle type, energy level, angle, and event number into parameter.txt========================
    case_index = 'primary is:' + str(primary) + ';           energy is: ' + str(energy) + ' Eev;          theta is: ' + str(
        e_theta) + ' degree;          phi is: ' + str(e_phi) + ' degree;          case num is:' + str(case)
    with open("..//result//" + target[i] + "//parameter.txt", 'w') as f:
        f.write(case_index)

        source_root = file_dir + '//antpos.dat'
        target_root = "..//result//" + target[i]
        shutil.copy(source_root,target_root)

        # ===================================Arbitrary input file first generates input parameters needed by subroutine================================================
        #  ================================Change according to actual situation==========================================
        # Select the galactic noise LST at the LST moment
        # lst = int(input("please input lst:"))
        # demo = int(input("please input demo number:"))
        lst = 18
        Ts = 0.5  # Manually enter the same time interval as the .trace file
        randomn = 0
        E_path = file_dir + '//a' + str(randomn) + '.trace'

        for a in os.listdir(file_dir):
            # print(root)
            # print(os.path.isdir(root))
            # if os.path.isdir(root) == True:            #判断是否为文件夹
            content.append(a)
        for b in content:
            # print(os.path.splitext(b)[1])
            if os.path.splitext(b)[1] == '.trace':
                if b.startswith('a') == True:
                    target_trace.append(b)

        #  ===========================start calculating===================
        [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(E_path, Ts, show_flag)  # Signal interception

        Edata = ex_cut
        Edata = np.column_stack((Edata, ey_cut))
        Edata = np.column_stack((Edata, ez_cut))

        [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(Edata, N, f1,
                                                                              show_flag)  # Frequency domain signal
        # =======Equivalent length================
        [Lce_complex, antennas11_complex_short] = CEL(e_theta, e_phi, N, f0, 1, show_flag)

        Lcehang = Lce_complex.shape[0]
        Lcelie = Lce_complex.shape[2]
        # ======Galaxy noise power spectrum density, power, etc.=====================
        [galactic_v_complex_double, galactic_v_time] = gala(lst, N, f0, f1, show_flag)
        # =================== LNA=====================================================
        [rou1_complex, rou2_complex, rou3_complex] = LNA_get(antennas11_complex_short, N, f0, 1, show_flag)
        # =======================  cable  filter VGA balun=============================================
        [cable_coefficient, filter_coefficient] = filter_get(N, f0, 1, show_flag)


# ===============================Start loop calculation========================================================================
    for num in range(len(target_trace)):
        # air shower,input file
        E_path = file_dir+ '//a' + str(num) + '.trace'
        xunhuan = '//a' + str(num) + '_trace.txt'

        # Output file path and name
        #  ===========================start calculating===========================================================
        [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = time_data_get(E_path, Ts, show_flag)  # Signal interception

        Edata = ex_cut
        Edata = np.column_stack((Edata, ey_cut))
        Edata = np.column_stack((Edata, ez_cut))

        [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(Edata, N, f1,
                                                                              show_flag)  # Frequency domain signal

        # =======Equivalent length================

        # ======Open circuit voltage of air shower=================
        Voc_shower_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
        # Frequency domain signal
        for p in range(Lcelie):
            Voc_shower_complex[:, p] = Lce_complex[:, 0, p] * E_shower_fft[:, 0] + Lce_complex[:, 1, p] * E_shower_fft[:,
                                                                                                          1] + Lce_complex[
                                                                                                               :, 2,
                                                                                                               p] * E_shower_fft[
                                                                                                                    :,
                                                                                                                    2] + 0
        # time domain signal
        [Voc_shower_t, Voc_shower_m_single, Voc_shower_p_single] = ifftget(Voc_shower_complex, N, f1, 2)

        # ======Galaxy noise power spectrum density, power, etc.=====================

        # ===========Voltage with added noise=======================================
        Voc_noise_t = np.zeros((Lcehang, Lcelie))
        Voc_noise_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
        for p in range(Lcelie):
            if noise_flag == 0:
                Voc_noise_t[:, p] = Voc_shower_t[:, p]
                Voc_noise_complex[:, p] = Voc_shower_complex[:, p]
            elif noise_flag == 1:
                Voc_noise_t[:, p] = Voc_shower_t[:, p] + galactic_v_time[random.randint(a=0, b=175), :, p]
                Voc_noise_complex[:, p] = Voc_shower_complex[:, p] + galactic_v_complex_double[random.randint(a=0, b=175), :, p]

        [Voc_noise_t_ifft, Voc_noise_m_single, Voc_noise_p_single] = ifftget(Voc_noise_complex, N, f1, 2)

        # ==================Voltage after LNA=====================================================
        V_LNA_complex = np.zeros((N, Lcelie), dtype=complex)
        for p in range(Lcelie):
            V_LNA_complex[:, p] = rou1_complex[:, p] * rou2_complex[:, p] * rou3_complex[:, p] * Voc_noise_complex[:, p] + 0
        [V_LNA_t, V_LNA_m_single, V_LNA_p_single] = ifftget(V_LNA_complex, N, f1, 2)

        # ======================Voltage after  cable=============================================
        V_cable_complex = np.zeros((N, Lcelie), dtype=complex)
        for p in range(Lcelie):
            V_cable_complex[:, p] = V_LNA_complex[:, p] * cable_coefficient[:, p] + 0
        [V_cable_t, V_cable_m_single, V_cable_p_single] = ifftget(V_cable_complex, N, f1, 2)

        # ======================Voltage after filter=============================================
        V_filter_complex = np.zeros((N, Lcelie), dtype=complex)
        for p in range(Lcelie):
            V_filter_complex[:, p] = V_LNA_complex[:, p] * cable_coefficient[:, p] * filter_coefficient[:, p] + 0
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
