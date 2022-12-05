#ccenv root
import grand.io.root_files as grf
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import optimize

# Analysis of 20Hz data
# Taken on Auger site in November 2022
# Using wrapper class from root_files
# OMH Nov 20229

def singleTraceProcess(t,x,N=5,DISPLAY=True):
   sig = np.std(x)
   th = N*sig
   ipk = abs(x)>th
   ttrig = t[ipk]  # Pulses above threshold
   xtrig = x[ipk]
   dttrig = np.diff(ttrig) # Time differences
   if len(xtrig)>0:
      dttrig = np.insert(dttrig, 0, [ttrig[0]]) # Insert time delay to beginning of trace
   itrue = dttrig>0.50 #mus
   fft=np.abs(np.fft.rfft(x))
   freq=np.fft.rfftfreq(len(x))*fsamp/1e6
   ntrig = sum(itrue)

   #show traces
   if ntrig>0:
       print("Event", i)
       print("Channel x: sigma = ",sig, "LSB, Nb spikes=",ntrig)
       if DISPLAY:
           #plt.subplots(2,1)
           #plt.subplot(211)
           plt.figure()
           plt.plot(t,x)
           plt.plot(ttrig[itrue],xtrig[itrue],'or')
           plt.plot([0,max(t)],[th,th],'--r')
           plt.plot([0,max(t)],[-th,-th],'--r')
           plt.xlim(min(t),max(t))
           plt.xlabel("Time (mus)")
           plt.ylabel("Channel 0 (LSB)")

           # FFT
           #plt.subplot(212)
           #plt.semilogy(freq,fft)
           #plt.xlim(min(freq),max(freq))
           #plt.xlabel("Frequency (MHz)")
           #plt.ylabel("log(FFT)")

           #plt.show()

   return ntrig, sig, fft, freq

DISPLAY = True
sites = {'BLS':[6010, 6080, 6081],'LosLeones':[6020],'Claire':[6030],'Culen':[6040],'AERA':[6050,6051],'AERApowerline':[6060,6061],'CLF':[6070,6071],'Mafalda':[6090,6091],'SolaStereo':[6100, 6101,6110,6111]}
N = 5

## Load data
f=sys.argv[1]  # FIle name
try:
    fa = grf.FileADCevent(f)
except:
    print("Could not load file",f,". Abort.")
    sys.exit()
try:
    adc = fa.get_obj_handlingtracesofevent_all()
except:
    print("Could not load ADCEvent tree. Abort.")  # Fails when trace size is not a constant+integer in the file
    sys.exit()
traces = adc.traces  #  Array of dimension nevents*nchannels*nsamples

runpath = f.split("/")[-1].split(".")[0]
runid = runpath.split("_")[0][2:]
runid = int(runid)
print("Run:",runid)
#Find site for this run
for i,s in enumerate(sites):
  if runid in list(sites.values())[i]:
      site = s
      break
  else:
      site = 'unknown'
print("Site=",site)
subrun=runpath.split("_")[-1]
print("Subrun:",subrun)
fsamp=500e6  # Hardcoded
print("Sampling frequency:",fsamp)
ib0 = traces.shape[2]  # Trace length. Warning: in raw data this can differ for first events (left overs??)
print("Trace length:",ib0)
nevents = traces.shape[0]
print("Nb of events in run:",nevents)
nch = traces.shape[1]
t = np.linspace(0,ib0/fsamp,ib0)*1e6

# Loop on events
ntrigs = 0
for i in range(min(nevents,1e200)):
    if i/100 == np.floor(i/100):
        print(i,"/",nevents)
    ntrig, sig, fft, freq = singleTraceProcess(t,traces[i,0,:],N,DISPLAY)
    ntrigs += ntrig
    try:
      meanfft = meanfft + fft
    except:
      meanfft = fft

#Summary stats
dur_ev = ib0/fsamp
dur_tot = nevents*dur_ev
trig_rate = ntrigs/dur_tot
print(ntrigs,"pulses above",N,"sigmas in",nevents,"timetraces of length",dur_ev*1e6,"mus.")
print("Mean = ", ntrigs/nevents,"pulses/trace, Rate = ",trig_rate,"Hz")
print('Total run duration is',dur_tot,'s.')
for i in range(nch):
  print("Std dev Channel",i,":",np.std(traces[:,i,:]))

# Summary histograms
xdata = traces[:,0,:].flatten()  # All data along x channel
meanfft = meanfft/nevents
meanfft[0:10] = meanfft[10] # Kill first frequency bins
plt.figure()
hamp = plt.hist(xdata, 200)
hampx = hamp[1][1:]
hampy = hamp[0]/dur_tot  # Normalize to time

plt.figure()
plt.subplots(2,1)
plt.subplot(211)
plt.plot(hampx,hampy,'+')
sig = np.std(xdata)
if DISPLAY:
    # Alternative: gauss fit
    p0 = [nevents*ib0/200,0,sig]
    print(p0)
    sel3s = np.logical_and(hampx>-3*sig,hampx<3*sig)
    sel5s = np.logical_and(hampx>-5*sig,hampx<5*sig)
    def gauss (x, x0,mu,sigma):
        p = [x0, mu,sigma]
        return p[0] * np.exp(-(x-p[1])**2 / (2 * p[2]**2))
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
    plt.xlabel('All amplitudes (LSB)')
    plt.title(runpath)
    plt.grid()

    plt.subplot(212)
    plt.semilogy(freq,meanfft)
    plt.xlim(0,max(freq))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('FFT')
    plt.savefig("figs/summary_"+runpath+".png")
if DISPLAY:
    plt.show()

hamp = [hampx, hampy]
sfreq = [freq, meanfft]
stats = [nevents,dur_tot,sig,trig_rate]
np.savez("npz/summary_"+runpath, hamp=hamp,sfreq=sfreq,site=site,stats=stats)
