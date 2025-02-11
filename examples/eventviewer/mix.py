import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, lfilter
import hdf5fileinout as hdf5io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################################

def get_filtered_peakAmpTime_Hilbert(InputFilename_, EventName_, AntennaInfo_, f_min_, f_max_):

    _NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo_)
    _peakamp, _peaktime = np.zeros(_NumberOfAntennas), np.zeros(_NumberOfAntennas)
    i = 0
    for ant_id in hdf5io.GetAntIDFromAntennaInfo(AntennaInfo_):

        _Efield_trace = hdf5io.GetAntennaEfield(InputFilename_,EventName_,ant_id,OutputFormat="numpy")
        _Efield_trace[:,0] *= 1.e-9 #from ns to s

        _Efield_filt = filters(_Efield_trace, FREQMIN=f_min_, FREQMAX=f_max_)

        _hilbert_amp = np.abs(hilbert(_Efield_filt[1:4,:]))
        _peakamp[i]=max([max(_hilbert_amp[0,:]), max(_hilbert_amp[1,:]), max(_hilbert_amp[2,:])])
        _peaktime[i]=_Efield_filt[0,np.where(_hilbert_amp == _peakamp[i])[1][0]]
                        
        i+=1

    return _peaktime, _peakamp


#Real antennes positions to airshower plane positions
def get_in_shower_plane(pos, k, core, inclination, declination):

    pos= (pos - core[:,np.newaxis])
    B  = np.array([np.sin(inclination)*np.cos(declination),
                   np.sin(inclination)*np.sin(declination),
                   np.cos(inclination)])
    kxB = np.cross(k,B)
    kxB /= np.linalg.norm(kxB)
    kxkxB = np.cross(k,kxB)
    kxkxB /= np.linalg.norm(kxkxB)
    #print("k_", k_, "_kxB = ", _kxB, "_kxkxB = ", _kxkxB)

    return np.array([np.dot(kxB, pos), np.dot(kxkxB, pos), np.dot(k, pos)])

def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')  # (order, [low, high], btype)

    return lfilter(b, a, data)


def filters(voltages, FREQMIN=50.e6, FREQMAX=200.e6):
  """ Filter signal v(t) in given bandwidth
  Parameters
  ----------
   : voltages
      The array of time (s) + voltage (muV) vectors to be filtered
   : FREQMIN
      The minimal frequency of the bandpass filter (Hz)
   : FREQMAX:
      The maximal frequency of the bandpass filter (Hz)


  Raises
  ------
  Notes
  -----
  At present Butterworth filter only is implemented
  Examples
  --------
  ```
  >>> from signal_treatment import _butter_bandpass_filter
  ```
  """

  t = voltages[:,0]
  v = np.array(voltages[:, 1:])  # Raw signal

  fs = 1 / np.mean(np.diff(t))  # Compute frequency step
  #print("Trace sampling frequency: ",fs/1e6,"MHz")
  nCh = np.shape(v)[1]
  vout = np.zeros(shape=(len(t), nCh))
  res = t
  for i in range(nCh):
        vi = v[:, i]
        #vout[:, i] = _butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs)
        res = np.append(res,_butter_bandpass_filter(vi, FREQMIN, FREQMAX, fs))

  res = np.reshape(res,(nCh+1,len(t)))  # Put it back inright format
  return res
