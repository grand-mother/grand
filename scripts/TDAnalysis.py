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
uid=int(sys.argv[2])
if len(sys.argv)>3:
    nshow=int(sys.argv[3])
else:
    nshow=0
print(">python TDAnalysis.py file unit_id [Nevents to show]")

evt=rt.ADCEventTree(f)
listevt=evt.get_list_of_events()
if nshow>len(listevt):
    nshow=len(listevt)

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
print("Nb shown:",nshow)
traceX=np.zeros((len(listevt),ib0),dtype='int')
traceY=np.zeros((len(listevt),ib1),dtype='int')
traceZ=np.zeros((len(listevt),ib2),dtype='int')

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
    return ntrigs, freq, fft

t = np.linspace(0,ib0/fsamp,ib0)*1e6
gpstime=np.zeros((len(listevt)),dtype='int')
tev = np.zeros((len(listevt)),dtype='int')
sig = np.zeros((len(listevt),3),dtype='float')
mfft = []
for i in range(nevents):
    evt.get_event(listevt[i][0],listevt[i][1])

    gpstime[i]=evt.gps_time[0]
    #print(evt.gps_time[0])
    ### How ugly is that????
    ind = np.argwhere( np.array(evt.du_id) == uid)
    if len(ind) == 0:  # tqrget ID not found in this event
      continue
    ind = int(ind[0][0])
    traceX[i]=np.array(evt.trace_0[ind])
    traceY[i]=np.array(evt.trace_1[ind])
    traceZ[i]=np.array(evt.trace_2[ind])

    #print("Event", i, "nCh:",np.shape(evt.trace_0)[0])
    if i/100 == int(i/100):
        print("Event", i, "Detection Unit",evt.du_id[ind],"Ch1:",np.std(traceX[i]),"Ch2:",np.std(traceY[i]),"Ch3:",np.std(traceZ[i]))

    # Get standard deviation
    tev[i] = evt.du_seconds[ind]
    sig[i] = [np.std(traceX[i]),np.std(traceY[i]),np.std(traceZ[i])]

    # Analyse traces
    ntrigsx, freqx, fftx = singleTraceProcess(traceX[i])
    ntrigsy, freqy, ffty = singleTraceProcess(traceY[i])
    ntrigsz, freqz, fftz = singleTraceProcess(traceZ[i])
    try:
        mfft = mfft + [fftx, ffty, fftz]
        ntrigs = ntrigs + [ntrigsx, ntrigsy,ntrigsz]
    except:
        mfft = np.array([fftx, ffty, fftz])
        ntrigs = np.array([ntrigsx, ntrigsy,ntrigsz])

sig = np.array(sig)
tev = np.array(tev)
tev = (tev-tev[0])/60
mfft = mfft/nevents
#Summary stats
#ntrigs = len(dttrigall)
dur_ev = ib0/fsamp
dur_tot = nevents*dur_ev
trig_rate = ntrigs/dur_tot
print(ntrigs,"pulses above",N,"sigmas in",nevents,"timetraces of length",dur_ev*1e6,"mus.")
print("Mean = ", ntrigs/nevents,"pulses/trace, Rate = ",trig_rate,"Hz")
print("Std dev Channel 0:",np.std(traceX))
print("Std dev Channel 1:",np.std(traceY))
print("Std dev Channel 2:",np.std(traceZ))

# Summary histograms
def gauss (x, x0,mu,sigma):
    p = [x0, mu,sigma]
    return p[0] * np.exp(-(x-p[1])**2 / (2 * p[2]**2))

plt.figure()
nbins = 1000
for i in range(3):
    plt.subplot(311+i)
    if i == 0:
        p0 = [0,0,np.std(traceX)/2]
        xdata = traceX.flatten()
    if i == 1:
        p0 = [0,0,np.std(traceY)/2]
        xdata = traceY.flatten()
    if i == 2:
        p0 = [0,0,np.std(traceZ)/2]
        xdata = traceZ.flatten()
    hamp = plt.hist(xdata, nbins)
    hampx = hamp[1][1:]
    hampy = hamp[0]
    p0[0] = max(hampy)
    print(p0)
    #gauss_ini = p0[0] * np.exp(-(hampx-p0[1])**2 / (2 * p0[2]**2))
    #plt.plot(hampx,gauss_ini,'k')  # Display in 5sigma range
    sel3s = np.logical_and(hampx>-1*np.mean(sig[:,i]),hampx<1*np.mean(sig[:,i]))
    sel5s = np.logical_and(hampx>-5*np.mean(sig[:,i]),hampx<5*np.mean(sig[:,i]))
    try:
        fit,pcov = optimize.curve_fit(gauss, hampx[sel3s], hampy[sel3s], p0=p0)  # Gauss fit in 3sigma range
        perr = np.sqrt(np.diag(pcov))
        parname = ['A','mu (LSB)','sig (LSB)']
        for ii in range(len(p0)):
            print(parname[ii],':',fit[ii],"+-",perr[ii])
        gauss_curve = fit[0] * np.exp(-(hampx-fit[1])**2 / (2 * fit[2]**2))
        plt.plot(hampx[sel5s],gauss_curve[sel5s],'k')  # Display in 5sigma range
    except:
        print("Gaussian fit of histogram has failed.")

    plt.yscale('log')
    plt.grid()
plt.xlabel('All amplitudes (LSB)')
tit = runpath + "- DU" + str(uid)
plt.subplot(311)
plt.title(tit)

lab = ["X channel","Y channel","Z channel"]
plt.figure()
for i in range(3):
    mfft[i,0:10] = mfft[i,10]
    #plt.subplot(311+i)
    plt.plot(freqx,mfft[i],label=lab[i])
    plt.xlim(0,max(freqx))
    plt.ylabel('FFT')
plt.xlabel('Frequency (MHz)')
plt.legend(loc="best")
plt.title(tit)

plt.figure()
for i in range(3):
    mfft[i,0:10] = mfft[i,10]
    #plt.subplot(311+i)
    plt.semilogy(freqx,mfft[i])
    plt.xlim(0,max(freqx))
    plt.ylabel('FFT')
plt.xlabel('Frequency (MHz)')
plt.legend(loc="best")
plt.title(tit)

plt.figure()
for i in range(3):
    plt.plot(tev,sig[:,i],".",label=lab[i])
plt.legend(loc="best")
plt.ylabel('Std Dev (LSB)')
plt.xlabel('Time (mn)')
tit = runpath + "- DU" + str(uid)
plt.title(tit)

plt.show()
