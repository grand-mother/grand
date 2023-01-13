#ccenv root
import grand.io.root_trees as rt
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import optimize
#from scipy import signal
from scipy.stats import norm

# Analysis of 20Hz data
# Taken on Auger site in NOvember 2022
# OMH Nov 2022

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
           plt.subplots(2,1)
           plt.subplot(211)
           plt.plot(t,x)
           plt.plot(ttrig[itrue],xtrig[itrue],'or')
           plt.plot([0,max(t)],[th,th],'--r')
           plt.plot([0,max(t)],[-th,-th],'--r')
           plt.xlim(min(t),max(t))
           plt.xlabel("Time (mus)")
           plt.ylabel("Channel 0 (LSB)")

           # FFT
           plt.subplot(212)
           plt.semilogy(freq,fft)
           plt.xlim(min(freq),max(freq))
           plt.xlabel("Frequency (MHz)")
           plt.ylabel("log(FFT)")

   return ntrig, sig, fft, freq

DISPLAY = True
sites = {'BLS':[6010, 6080, 6081],'LosLeones':[6020],'Claire':[6030],'Culen':[6040],'AERA':[6050,6051],'AERApowerline':[6060,6061],'CLF':[6070,6071],'Mafalda':[6090,6091],'SolaStereo':[6100, 6101,6110,6111]}
N = 5
#args are rootfile and number of events to show
f=sys.argv[1]
if len(sys.argv)>2:
    nshow=int(sys.argv[2])
else:
    nshow=0

evt=rt.ADCEventTree(f)
listevt=evt.get_list_of_events()
if nshow>len(listevt):
    nshow=len(listevt)

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
evt.get_event(listevt[10][0],listevt[10][1]) # Skip first 3 events
fsamp=evt.adc_sampling_frequency[0]*1e6 #Hz
#assuming the same configuration for all events of the run
ib0=evt.adc_samples_count_channel0[0]
ib1=evt.adc_samples_count_channel1[0]
ib2=evt.adc_samples_count_channel2[0]
ib3=evt.adc_samples_count_channel3[0]
print("Sampling frequency:",fsamp)
print("Trace length:",ib0)
nevents = np.shape(listevt)[0]
print("Nb of events in run:",nevents)

t = np.linspace(0,ib0/fsamp,ib0)*1e6

gpstime=np.zeros((len(listevt)),dtype='int')
max0=np.zeros((len(listevt)),dtype='int')
trace0=np.zeros((len(listevt),ib0),dtype='int')
trace1=np.zeros((len(listevt),ib1),dtype='int')
trace2=np.zeros((len(listevt),ib2),dtype='int')
trace3=np.zeros((len(listevt),ib3),dtype='int')
ntrigs = 0


for i in range(min(nevents,1e200)):
    evt.get_event(listevt[i][0],listevt[i][1])
    #print("event length=",evt.adc_samples_count_channel0[:])
    #print(type(evt.adc_samples_count_channel0))
    if i<10:  # Skip first events because could be leftovers in memory
        continue

    gpstime[i]=evt.gps_time[0]
    #print(evt.gps_time[0])
    #print(evt.gps_lat[0])
    trace0[i]=evt.trace_0[0]
    trace1[i]=evt.trace_1[0]
    trace2[i]=evt.trace_2[0]
    trace3[i]=evt.trace_3[0]

    if i/100 == np.floor(i/100):
      print(i,"/",nevents)
    ntrig, sig, fft, freq = singleTraceProcess(t,trace0[i],N,DISPLAY)
    ntrigs += ntrig
    if ntrig>0:
        #plt.show()
        #print("Nb spikes observed:",ntrig)
        a = 1
    try:
      meanfft = meanfft + fft
    except:
      meanfft = fft

#np.savetxt("txt/" + runpath + ".txt", dttrigall)

#Summary stats
dur_ev = ib0/fsamp
dur_tot = nevents*dur_ev
trig_rate = ntrigs/dur_tot
print(ntrigs,"pulses above",N,"sigmas in",nevents,"timetraces of length",dur_ev*1e6,"mus.")
print("Mean = ", ntrigs/nevents,"pulses/trace, Rate = ",trig_rate,"Hz")
print('Total run duration is',dur_tot,'s.')
print("Std dev Channel 0:",np.std(trace0))
#print("Std dev Channel 1:",np.std(trace1))

# Summary histograms
xdata = trace0.flatten()
meanfft = meanfft/nevents
meanfft[0:10] = meanfft[10]

plt.subplots(2,1)
plt.subplot(211)
hamp = plt.hist(xdata, 100)
hampx = hamp[1][1:]
hampy = hamp[0]/dur_tot  # Normalize to time
sig = np.std(xdata)
if DISPLAY:
    # Alternative: gauss fit
    p0 = [nevents*ib0/100,0,sig]
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
