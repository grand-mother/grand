"""! Signal processing

- This module contains several signal processing
functionalities to be applied to simulation/data
- operations are meant to be on the signal traces 
for individual antennas, suitable to be used both
in Grandlib format/ read from hdf5 files
- expects signal traces to be of the size (3,lengthoftrace)
"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, resample, decimate, butter, lfilter

logger = getLogger(__name__)


def get_filter(time, trace, fr_min, fr_max):
    """!Filter signal  in given bandwidth
    @note
      At present Butterworth filter only is implemented, others: what
      is close to hardware filter?

    @param time (array): [ns] time
    @param trace (array): ElectricField (muV/m)/voltage (muV) vectors to be filtered
    @param fr_min (float): [Hz] The minimal frequency of the bandpass filter
    @param fr_max (float): [Hz] The maximal frequency of the bandpass filter

    @returns: filtered trace in time domain
    """
    tstep = (time[1] - time[0]) * 1e-09  # s
    rate = 1 / tstep
    nyq = 0.5 * rate  # Nyquist limit
    low = fr_min / nyq
    high = fr_max / nyq
    order = 5
    coeff_b, coeff_a = butter(order, [low, high], btype="band")
    filtered = lfilter(coeff_b, coeff_a, trace)  # this is data in the time domain
    return filtered


def get_fft(time, trace, specialwindow=False):
    """!Compute the one-dimensional discrete Fourier Transform for real input
    @note
      - numpy rfft pack calculates only positive frequency parts exploiting Hermitian-Symmetry,
        the returned array is complex and of length N/2+1
      - for plotting purposes taking  abs(fft trace) is required
      - for now a hamming window is used, special windows help with subtle effects like signal
        leakage,might be more important when used on measured data

    @param time (array): [ns] time
    @param trace (array) : ElectricField (muV/m)/voltage (muV) vectors to be filtered, could be raw or
      filtered
    @param specialwindow (bool): if true implements a hamming window

    @returns (array): [Mhz]frequency , fft trace(complex)
    """
    dlength = trace.shape[1]  # length of  the trace
    tstep = (time[1] - time[0]) * 1e-9  # time step in sec
    freq = np.fft.rfftfreq(dlength, tstep) / 1e06  # MHz
    if not specialwindow:
        trace_fft = np.fft.rfft(trace)
    else:
        logger.debug("performing a hamming window on the signal ...")
        hammingwindow = np.hamming(dlength)
        trace_fft = np.fft.rfft(trace * hammingwindow)
    return freq, trace_fft


def get_inverse_fft(trace):
    """!Computes the inverse of rfft

    @param trace (array): signal trace same as in get_fft

    @return (array): inverse fft (time domain)
    """
    inv_fft = np.fft.irfft(trace, n=trace.shape[1])
    return inv_fft


def get_peakamptime_hilbert(time, trace, f_min, f_max, filtered=False):
    """!Get Peak and time of EField trace, either filtered or unfiltered

    @param time (array): time
    @param trace (array): ElectricField (muV/m) vectors to be filtered- expected
      size (3,dlength), dlength= size of the signal trace
    @param f_min (float): [Hz] The minimal frequency of the bandpass filter
    @param f_max (float): [Hz] The maximal frequency of the bandpass filter
    @param filtered (bool): if true filtering is applied else raw input trace is used

    @return TBD
    """
    if filtered:
        logger.debug("filtering the signal .....")
        filt = get_filter(time, trace, f_min, f_max)
    else:
        logger.debug("continuing with raw signal ...")
        filt = trace

    hilbert_amp = np.abs(hilbert(filt, axis=-1))
    peakamp = max([max(hilbert_amp[0]), max(hilbert_amp[1]), max(hilbert_amp[2])])
    peaktime = time[np.argwhere(hilbert_amp == peakamp)[0][1]]

    return peaktime, peakamp


def digitize_signal(time, trace, tsampling, downsamplingmethod=1):

    """!Performs digitization/resampling of signal trace for a given sampling rate
    @note
      There are several ways of resampling- scipy.resample seems most
      common for up/down sampling, an extra method is also added for
      downsampling-scipy.decimate - this seems to use antialiasing filter.
      these methods might change in future.

    @param time (array): [ns] time trace
    @param trace (array):  signal trace - efield(meuV/m)/Voltage(meuV/m)
    @param tsampling (integer/float): [ns] desired sampling rate in time
    @param downsamplingmethod (integer):  1 for scipy.resample, 2.scipy.decimate
    @return resampled signal and time trace
    """

    tstep = time[1] - time[0]  # ns
    samplingrate = tsampling / tstep
    dlength = trace.shape[1]

    if samplingrate < 1:
        logger.debug("performing umsampling ...")
        resampled = [resample(x, round(1 / samplingrate) * dlength) for x in trace]
        time_resampled = np.linspace(time[0], time[-1], len(resampled[0]), endpoint=False)
    else:
        if downsamplingmethod == 1:
            logger.debug("performing downsampling method 1 ...")
            resampled = [resample(x, dlength / round(samplingrate)) for x in trace]

        elif downsamplingmethod == 2:
            logger.debug("performing downsampling with method 2 ...")
            resampled = [decimate(x, round(samplingrate)) for x in trace]
        else:
            raise ValueError("choose a valid method number")
        time_resampled = np.linspace(time[0], time[-1], len(resampled[0]), endpoint=False)
    resampled = np.array(resampled).reshape(3, len(resampled[0]))

    return resampled, time_resampled


def add_noise(trace, vrms):
    """!Add normal random noise on traces

    @todo
      Later should be modified to add measured noise

    @param trace (array): voltage trace
    @param vrms (float): noise rms, e.g vrmsnoise= 15 meu-V

    @return (array): noisy voltage trace
    """

    noise = np.random.normal(0, vrms, size=trace.shape[1])
    noisy_traces = np.add(trace, noise)

    return noisy_traces


def complex_expansion(N, f0, f1, f2, data):
    """!
    # This Python file uses the following encoding: utf-8

    # = == == == == This procedure is used as a subroutine to complete the expansion of the spectrum == == == == =
    # % N is the number of frequency points, that is, the spectrum that needs to be expanded
    # % f0 is the frequency step, MHz
    # % f1 is the starting frequency of the spectrum to be expanded, f2 is the cutoff frequency of the spectrum to be expanded
    # % The program only considers that the length of the expanded data is less than floor(N / 2), such as N = 10, the length of the expanded data <= 5; N = 9, the length of the expanded data <= 4
    # data 1 dimension
    """
    
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
        data_expansion[f_hang_start : f_hang_end + 1] = data
        data_expansion[duichen] = data[1:].conjugate()
    else:
        a1 = np.arange(0, f_hang_start - 1 + 1, 1).tolist()
        a2 = np.arange(f_hang_end + 1, N - f_hang_start - effective + 1, 1).tolist()
        a3 = np.arange(N - f_hang_start + 1, N, 1).tolist()  # Need to make up 0;
        add = a1 + a2 + a3
        add = np.array(add)
        duichen = np.arange(N - f_hang_start, N - f_hang_start - effective, -1)
        data_expansion[add] = 0
        data_expansion[f_hang_start : f_hang_end + 1] = data[:]
        data_expansion[duichen] = data.conjugate()

    return f, data_expansion

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

        data_fft_m_single = data_fft_m[0 : len(f1)]  # unilateral

        data_fft_p = np.angle(data_fft, deg=True)  # phase
        data_fft_p = np.mod(data_fft_p, 2 * 180)
        # data_fft_p_deg = np.rad2deg(data_fft_p)
        data_fft_p_single = data_fft_p[0 : len(f1)]

    string = np.array(["x", "y", "z"])
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        for j in range(lienum):
            plt.rcParams["font.sans-serif"] = ["Times New Roman"]
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
            plt.rcParams["font.sans-serif"] = ["Times New Roman"]
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
            data_ori_m_single = data_ori_m[0 : len(f1)]  # unilateral

            data_ori_p[:, i] = np.angle(data_ori[:, i], deg=True)  # phase
            data_ori_p[:, i] = np.mod(data_ori_p[:, i], 2 * 180)
            data_ori_p_single = data_ori_p[0 : len(f1)]

    elif true == 2:
        for i in range(lienum):
            data_ori_m[:, i] = abs(data_ori[:, i]) * 2 / N
            data_ori_m[0] = data_ori_m[0] / 2

            data_ori_m_single = data_ori_m[0 : len(f1)]  # 单边

            data_ori_p = np.angle(data_ori, deg=True)  # 相位
            data_ori_p = np.mod(data_ori_p, 2 * 180)  # -pi到pi转为0到2pi
            data_ori_p_single = data_ori_p[0 : len(f1)]

    # % % 时域
    data_ifft = np.zeros((N, lienum))
    for i in range(lienum):
        data_ifft[:, i] = ifft(data_ori[:, i]).real

    return np.array(data_ifft), np.array(data_ori_m_single), np.array(data_ori_p_single)


