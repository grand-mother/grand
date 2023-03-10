from grand.io.root_files import File
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
params = {
    "legend.fontsize": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.figsize": (10, 8),
    "axes.grid": False,
}
plt.rcParams.update(params)


#V_JM = File("/home/grand/scripts/c2_merge_test.root")
V_mer = File("/home/data_challenge1_pm_lwp/data/Coarse2_xmax_add_voltage_event_PC.root")
V_LP  = File("/home/data_challenge1_pm_lwp/data/Coarse2_xmax_add_voltage_event.root")
#V_LP = File("/home/data_challenge1_pm_lwp/results_external_PM_withNoise_withRF/Coarse2_xmax_add_voltage.root")

V_mer.get_event(event_idx=0)
V_LP.get_event(event_idx=0)

#V_JM.get_next_event()
#V_LP.get_next_event()

print(V_mer.traces.shape, V_LP.traces.shape)
idx_du=44

xmin, xmax = 70, 250
ymin, ymax = -1000, 1000
a1, a2 = 0, 2000
b1, b2 = 0, 2000

plt.figure()
# Using Leff in antenna frame for convolving with Efield rfft.
plt.subplot(311)
plt.plot(V_mer.traces[idx_du,0], color='r', label='PC')
plt.plot(V_LP.traces[idx_du,0], color='b', label='LP')

plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
#plt.xticks(ticks=np.arange(0,1001,100), labels=[])
#plt.text(200, 0.7*np.max(V_JM_antframe.traces[idx_du]),'JM (Leff ant frame)', fontsize=20, color='k')
#plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)

plt.subplot(312)
plt.plot(V_mer.traces[idx_du,1], color='r', label='PC')
plt.plot(V_LP.traces[idx_du,1], color='b', label='LP')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.legend(loc='upper right')
plt.grid(ls='--', alpha=0.3)
#plt.xticks(ticks=np.arange(0,1001,100), labels=[])
#plt.text(200, 0.7*np.max(V_LP.traces[idx_du]),'LP', fontsize=20, color='k')
#plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)

plt.subplot(313)
plt.plot(V_mer.traces[idx_du,2], color='r', label='PC')
plt.plot(V_LP.traces[idx_du,2], color='b', label='LP')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
#plt.legend(loc='upper right')
plt.grid(ls='--', alpha=0.3)
#plt.text(200, 0.7*np.max(V_LP.traces[idx_du]-V_JM_antframe.traces[idx_du]),'LP-JM_ant_frame', fontsize=20, color='k')
#plt.xlim(xmin, xmax)
#plt.ylim(-10,10)
#plt.xlabel('Time bins')
plt.savefig("/home/grand/scripts/merge_test_voltage.png")
plt.show()
