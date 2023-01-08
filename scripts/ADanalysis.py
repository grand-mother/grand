#ccenv root
import grand.io.root_trees as rt
import matplotlib.pyplot as plt
import sys
import numpy as np
#from scipy import optimize
#from scipy import signal
from scipy.stats import norm

# Analysis of 10s data @ Nancay
# July 2022 OMH

showhisto=1
dttrigall = [];
#args are rootfile and number of events to show
f=sys.argv[1]
print(">python ADAnalysis.py")

evt=rt.ADCEventTree(f)
listevt=evt.get_list_of_events()

runpath = f.split("/")[-1].split(".")[0]
runid=runpath.split("_")[0]
print("Run:",runid)
subrun=runpath.split("_")[-1]
print("Subrun:",subrun)
evt.get_event(listevt[0][0],listevt[0][1])
fsamp=evt.adc_sampling_frequency[0]*1e6 #Hz
#assuming the same configuration for all events of the run
ib0=evt.adc_samples_count_channel0[0]
ib1=evt.adc_samples_count_channel1[0]
ib2=evt.adc_samples_count_channel2[0]
print("Sampling frequency:",fsamp)
print("Trace length:",ib0)
nevents = np.shape(listevt)[0]
print("Nb of events in run:",nevents)

N = 6
def singleTraceProcess(s):
    th = N*np.std(s)
    ipk = s>th
    ttrig = t[ipk]  # Pulses above threshold
    xtrig = s[ipk]
    dttrig = np.diff(ttrig) # Time differences
    if len(xtrig)>0:
        dttrig = np.insert(dttrig, 0, [ttrig[0]]) # Insert time delay to beginning of trace
    itrue = dttrig>0.20 #
    #print(ttrig,dttrig,itrue)
    #dttrigall = np.append(dttrigall,np.diff(ttrig[itrue]))
    ntrigs = sum(itrue)
    fft=np.abs(np.fft.rfft(s))
    freq=np.fft.rfftfreq(len(s))*fsamp/1e6
    if disp:
        plt.subplots(2,2)
        plt.subplot(221)
        plt.title(lab)
        plt.plot(t,s)
        plt.plot(ttrig[itrue],xtrig[itrue],'or')
        plt.plot([0,max(t)],[th,th],'--r')
        plt.xlim(min(t),max(t))
        plt.xlabel("Time (mus)")
        plt.ylabel("Channel 0 (LSB)")

        # Delta time pulses
        plt.subplot(223)
        plt.hist(dttrig[itrue],100)
        plt.xlabel("$\Delta$ t Pulses (mus)")

        # FFT
        plt.subplot(222)
        plt.semilogy(freq,fft)
        plt.xlim(min(freq),max(freq))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("log(FFT)")
        plt.subplot(224)
        plt.plot(freq,fft)
        plt.xlim(min(freq),max(freq))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("FFT")

        plt.show()

    return ntrigs, freq, fft

t = np.linspace(0,ib0/fsamp,ib0)*1e6
gpstime=np.zeros((len(listevt)),dtype='int')
for i in range(nevents):
    evt.get_event(listevt[i][0],listevt[i][1])
    #print("DU counts=",evt.du_count) # Error here...
    print("DUs in event:",evt.du_id)
    nDUs = len(evt.du_id)
    gpstime[i]=evt.gps_time[0]
    #print(evt.gps_time[0])
    for i in range(nDUs):
        plt.figure()
        traceX=np.array(evt.trace_0[i])
        traceY=np.array(evt.trace_1[i])
        traceZ=np.array(evt.trace_2[i])
        plt.subplot(311)
        plt.plot(t,traceX)
        plt.xlabel("Time (ns)")
        plt.ylabel("Channel X (LSB)")
        plt.title("DU"+str(evt.du_id[i]))
        plt.subplot(312)
        plt.plot(t,traceY)
        plt.xlabel("Time (ns)")
        plt.ylabel("Channel Y (LSB)")
        plt.subplot(313)
        plt.plot(t,traceZ)
        plt.xlabel("Time (ns)")
        plt.ylabel("Channel Z (LSB)")
    plt.show()
