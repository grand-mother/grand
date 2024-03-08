import grand.dataio.root_files as froot
import matplotlib.pyplot as plt
import numpy as np
import sys

f_event = sys.argv[1]
print(f_event)

ef3d=froot.get_handling3dtraces(f_event)
t_max , v_max =ef3d.get_tmax_vmax()

plt.figure()
plt.title(f_event)
t_tot = ef3d.get_size_trace()*ef3d.get_delta_t_ns()[0]
t_max_rel = t_max-ef3d.t_samples[:,0]
plt.scatter(100*t_max_rel/t_tot, v_max)
plt.xlim([0,100])
plt.xlabel("Position of time of max in trace, % ")
plt.ylabel(r"Max value of trace, $\mu$v/m ")
plt.hlines(25, 0, 100,label="level of galaxy background",linestyles=  '-')
v_inf  = np.min(v_max)/2
v_sup  = np.max(v_max)
plt.vlines(15, v_inf, v_sup,label="inf border for max",linestyles= '-.')
plt.vlines(55, v_inf, v_sup,label="sup border for max", linestyles= '--')
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()
plt.ion()
