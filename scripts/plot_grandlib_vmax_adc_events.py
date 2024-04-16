"""
Created on 16 avr. 2024

@author: jcolley
"""
import sys


import grand.dataio.root_trees as groot
import numpy as np 
import matplotlib.pyplot as plt


pnf_event = sys.argv[1]
s_evt = pnf_event.split("/")

# Read only traces
fevt = groot.TADC(pnf_event)
# close doesn't exist ?
#fevt.close()
print(fevt.event_number)
sys.exit()


a_nb_tr = ak.num(traces, axis=1)
nb_traces = ak.sum(a_nb_tr)
print("nb traces: ",nb_traces)
a_vmax = np.zeros(nb_traces, dtype=np.float32)
nb_evt = ak.num(traces, axis=0)
idx_t = 0
for idx, nb_tr in zip(range(nb_evt),a_nb_tr) :
    tr3d = ak.to_numpy(traces[idx])
    max_norm = np.max(np.linalg.norm(tr3d, axis=1), axis=1)
    a_vmax[idx_t : idx_t + nb_tr] = max_norm
    idx_t += nb_tr
print("end")

_, bins = np.histogram(a_vmax, bins=20)
logbins = np.logspace(0,np.log10(bins[-1]),len(bins))
plt.rcParams["savefig.bbox"]='tight'
plt.rcParams["savefig.pad_inches"]=0.2
plt.title(f"Histogram of max value for {nb_traces} traces")
plt.hist(a_vmax, bins=logbins)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(f"ADU\nDir: {s_evt[-2]}\nFile: {s_evt[-1]}")
plt.ylabel("Number of trace in bin")
plt.grid()
plt.savefig(f'histo_grandlib_{nb_traces}_traces.png')

#plt.show()