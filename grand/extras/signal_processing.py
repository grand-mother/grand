#!/usr/bin/env python3

""" Notes: 
    -This module contains several signal processing
    	functionalities to be applied to simulation/data
    - operations are meant to be on the signal traces 
        for individual antennas, suitable to be used both
        in Grandlib format/ read from hdf5 files
    - expects signal traces to be of the size (3,lengthoftrace)
"""


import numpy as np
from scipy.signal import hilbert, resample,decimate
from scipy.signal import butter, lfilter




################################################################################################3


def get_filter(time, trace, fr_min, fr_max):

    """ Filter signal  in given bandwidth
      Parameters
      ----------
       :  trace
         ElectricField (muV/m)/voltage (muV) vectors to be filtered
       : time
            Array of time in ns

       : fr_min
          The minimal frequency of the bandpass filter (Hz)
       : fr_max:
          The maximal frequency of the bandpass filter (Hz)
      returns:
       filtered trace in time domain
      Notes
      -----
      At present Butterworth filter only is implemented, others: what is close to hardware filter?


      """
    tstep = (time[1] - time[0]) * 1e-09  # s
    rate = 1 / tstep
    nyq = 0.5 * rate  # Nyquist limit
    low = fr_min / nyq
    high = fr_max / nyq
    order = 5
    coeff_b, coeff_a = butter(order, [low, high], btype='band')
    filtered = lfilter(coeff_b, coeff_a, trace)  # this is data in the time domain
    return filtered


def get_fft(time, trace, specialwindow=False):

    """ Compute the one-dimensional discrete Fourier Transform for real input.
          Parameters
          ----------
           :  trace
             ElectricField (muV/m)/voltage (muV) vectors to be filtered, could be raw or filtered
           : time
                Array of time in ns

           : specialwindow: Boolean
                    if true implements a hamming window
           -----------------------------
           returns:
                array : frequency(MHz) , fft trace(complex)
           ! Note:
               - numpy rfft pack calculates only positive frequency parts exploiting Hermitian-Symmetry, 
               the returned array is complex and of length N/2+1
               - for plotting purposes taking  abs(fft trace) is required
               - for now a hamming window is used, special windows help with subtle effects like signal leakage,
                might be more important when used on measured data

           """

    dlength = trace.shape[1]  # length of  the trace
    tstep = (time[1] - time[0]) * 1e-9  # time step in sec
    freq = np.fft.rfftfreq(dlength, tstep) / 1e06  # MHz
    if not specialwindow:
        trace_fft = np.fft.rfft(trace)
    else:
        print ('performing a hamming window on the signal ...')
        hammingwindow = np.hamming(dlength)
        trace_fft = np.fft.rfft(trace * hammingwindow)

    return freq, trace_fft


def get_inverse_fft(trace):

    """"
    Computes the inverse of rfft.
    ---------------------------
    parameters: signal trace same as in get_fft
    returns: inverse fft (time domain)


    """

    inv_fft = np.fft.irfft(trace, n=trace.shape[1])
    
    return inv_fft


def get_peakamptime_hilbert(time, trace, f_min, f_max, filtered=False):

    """ Get Peak and time of EField trace, either filtered or unfiltered
    Parameters
    ----------
    : Time
    : trace
        ElectricField (muV/m) vectors to be filtered- expected size (3,dlength), dlength= size of the signal trace
    : f_min
      The minimal frequency of the bandpass filter (Hz)
    : f_max:
      The maximal frequency of the bandpass filter (Hz)
    : filtered: boolean
   	    if true filtering is applied else raw input trace is used  """

    if filtered:
        print ('filtering the signal .....')
        filt = get_filter(time,trace, f_min, f_max)
    else:
        print('continuing with raw signal ...')
        filt = trace

    hilbert_amp = np.abs(hilbert(filt, axis=-1))
    peakamp = max([max(hilbert_amp[0]), max(hilbert_amp[1]), max(hilbert_amp[2])])
    peaktime =  time[np.argwhere(hilbert_amp == peakamp)[0][1]]

    return peaktime, peakamp


def digitize_signal(time,trace, tsampling ,downsamplingmethod=1):

    """
    performs digitization/resampling of signal trace for a given  sampling rate
    :param trace: (array) signal trace - efield(meuV/m)/Voltage(meuV/m)
    :param time: (array) time trace (ns)
    :param tsampling: (integer/float) desired sampling rate in time (ns)
    :param downsamplingmethod: (integer) 1 for scipy.resample, 2.scipy.decimate
    :return: resampled signal and time trace
    !Note: there are several ways of resampling- scipy.resample seems most common for up/down sampling,
    an extra method is also added for downsampling-scipy.decimate - this seems to use antialiasing filter.
    these methods might change in future.
    """

    tstep=time[1]-time[0] # ns
    samplingrate= tsampling/tstep
    dlength=trace.shape[1]
    
    if samplingrate<1:
        print ('performing umsampling ...')
        resampled= [resample(x,round(1/samplingrate)*dlength) for x in trace]
        time_resampled=np.linspace(time[0],time[-1],len(resampled[0]),endpoint=False)
    else:
        if downsamplingmethod==1:
            print ('performing downsampling method 1 ...')
            resampled = [ resample(x, dlength/round(samplingrate)) for x in trace]

        elif downsamplingmethod==2:
            print ('performing downsampling with method 2 ...')
            resampled = [decimate(x,round(samplingrate)) for x in trace]
        else:
            raise ValueError("choose a valid method number")
        time_resampled = np.linspace(time[0],time[-1],len(resampled[0]),endpoint=False)
    resampled = np.array(resampled).reshape(3,len(resampled[0]))
    
    return resampled,time_resampled


def add_noise(trace, vrms):
    """Add normal random noise on traces
    Parameters:
    -----------
        VTrace: numpy array
            voltage trace
        vrms: float
            noise rms, e.g vrmsnoise= 15 meu-V
    Returns:
    ----------
        numpy array
        noisy voltage trace
    !Note: Later should be modified to add measured noise
    """

    noise = np.random.normal(0, vrms, size=trace.shape[1])
    noisy_traces = np.add(trace, noise)

    return noisy_traces
