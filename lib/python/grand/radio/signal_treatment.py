"""
Astropy units missing
"""


import numpy as np

import logging
logger = logging.getLogger("Signal_Treatment")

#===========================================================================================================
def p2p(trace):
    ''' Calculates peak to peak values
    
    Arguments:
    ----------
    trace: np.array with 1 or 4 columns
            signal trace
    Returns:
    --------
    list: floats
        peak-to-peak values
    '''        
    
    if trace.ndim==2: # simply assume np.array([t, x, y, z])
        xy = np.sqrt(trace.T[1]**2 + trace.T[2]**2)
        combined = np.sqrt(trace.T[1]**2 + trace.T[2]**2 + trace.T[3]**2) 
        return max(trace.T[1])-min(trace.T[1]), max(trace.T[2])-min(trace.T[2]), max(trace.T[3])-min(trace.T[3]), max(xy)-min(xy), max(combined)-min(combined)
    elif trace.ndim==1:
        return max(trace)-min(trace)
    else:
        print("in p2p(): dimensions not correct")

#===========================================================================================================
def hilbert_env(signal):
    ''' 
    Hilbert envelope - abs(analytical signal)
    Arguments:
    ----------
    signal: np.array
        signal trace
    Returns:
    --------
    amplitude_envelope: np.array
        Hilbert envelope
        
    '''
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

#===========================================================================================================
def hilbert_peak(time, signal):
    ''' Calculates time and amplitude of peak
    
    Arguments:
    ----------
    time: numpy array
        time in ns
    signal: numpy array
        signal trace in muV or muV/m
        
    Returns:
    --------
    time of maximum in ns : float
    maximum of Hilbert envelope in muV or muV/m: float

    '''
    envelope=hilbert_env(signal)
    #Get time and amp
    
    return time[np.where(signal == signal.max())][0],max(envelope)


#===========================================================================================================
# --------- Trigger modules
def _trigger(p2p, mode, thrs):
    '''
    There are three modes: any, xy and all.
    According to each mode, returns 1 if the antenna is triggered or not.
    
    Arguments:
    ----------
    p2p: numpy array
        peak to peak values in muV/m or muV: x,y,z (,x+y, x+y+z combined)
    mode: str
        any, xy or all
    thrs: float
        threshold value to exceed in muV/m or muV
        
    Returns:
    ----------
    trig: int
        0 or 1
    '''
    
    if type(p2p) is list:
        p2p = np.array(p2p)


    if mode == 'any': #there are three modes: any, xy and all
        if p2p[0] >= thrs:
           return  1
        elif p2p[1] >= thrs:
           return  1
        elif p2p[2] >= thrs:
           return  1
        else:
           return 0    
    #elif mode == 'xy':
        #if p2p[0] >= thrs and p2p[1] >= thrs:
           #return 1
        #else:
           #return 0    
    #elif mode == 'all':
        #if p2p[0] >= thrs and p2p[1] >= thrs and p2p[2] >=thrs:
           #return 1
        #else:
           #return 0
    elif mode == 'xy':
        if p2p[3] >= thrs:
           return 1
        else:
           return 0    
    elif mode == 'all':
        if p2p[4] >=thrs:
           return 1
        else:
           return 0
    else:
       print("this mode doesn't exist")
    #return trig

#===========================================================================================================

# --------- Different SNR definitions
