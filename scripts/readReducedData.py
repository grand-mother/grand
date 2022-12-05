import sys
import glob
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def fetchSite(target_site):
    site_files = []
    print("Scanning reduced data files for site",target_site)
    dir = "/home/soft/grand/scripts/npz/"  # Loopng on all files
    files = glob.glob(dir+"/*.npz")
    for f in files:
        npzfile = np.load(f,allow_pickle=True)
        if npzfile['site'] == target_site:
            site_files.append(f)
    print("Files for site",target_site)
    print(site_files)
    return site_files

if __name__ == '__main__':
    files = []
    #files = files + fetchSite('LosLeones')
    #files = files + fetchSite('BLS')
    files = files + fetchSite('CLF')
    #files = files + fetchSite('AERA')
    #files = files + fetchSite('Mafalda')
    #files = files + fetchSite('SolaStereo')

    for f in files:
        #print("Loading file",f)
        npzfile = np.load(f,allow_pickle=True)
        # Build run name
        a = f.split("summary_")[1].split(".")[0]
        runid = int(a.split("_")[0][2:])
        subrunid = int(a.split("_")[1][1:5])
        #if subrunid == 1: # Skip runs with LNA off
        #    continue
        if runid == 6081: # Skip run 6081
            continue
        if npzfile['stats'][1] < 1e-2: # Skip runs with less than 1ms of data
            continue
        rname = str("R" + str(runid) + "." + str(subrunid))
        print("Run:",rname, "at site",npzfile['site'])
        #print("Stats:")
        print(int(npzfile['stats'][0]),"events recorded in",npzfile['stats'][1],"s")
        print("Std dev:",npzfile['stats'][2],"LSB")
        print("Spike rate:",npzfile['stats'][3],"Hz")

        hamp = npzfile['hamp']
        hampx = hamp[0,:]
        hampy = hamp[1,:]
        plt.figure(1)
        plt.semilogy(hampx,hampy,'+-',label= rname + ' (' + str(npzfile['site']) + ')')
        sfreq = npzfile['sfreq']
        freq = sfreq[0]
        spec = sfreq[1]
        plt.legend()
        plt.xlabel('ADC value (LSB)')
        plt.ylabel('Occurence (Hz)')

        plt.figure(2)
        plt.semilogy(freq,spec,label= rname + ' (' + str(npzfile['site']) + ')')
        plt.legend()
        plt.xlim([0, 250])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('FFT')

plt.show()
