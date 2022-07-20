#! /usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt 

path_data = sys.argv[1]
simu_tag = path_data.split('/')[-1]
print(simu_tag)
a_trace = np.loadtxt(path_data)

plt.figure()

plt.title(simu_tag)
plt.plot(a_trace[:, 0], a_trace[:, 1], label="col 1")
plt.legend()
plt.xlabel("time [ns] ?")
plt.ylabel("Volt ?")
plt.grid()

plt.show()
