#!/usr/bin/env python3

"""
Extract global TF of RF chain.

Nov. 2024, Colley Jean-Marc
"""

import numpy as np
import matplotlib.pyplot as plt

import grand.sim.detector.rf_chain as grfc


rfchain = grfc.RFChain()
freq_MHz = np.linspace(30, 251, 251 - 30 + 1)
print(freq_MHz)
rfchain.compute_for_freqs(freq_MHz)
print("Read RF")
rfc = rfchain.get_tf()
rfc2 = rfchain.get_tf()
rfc2 = rfchain.get_tf()
rfc2 = rfchain.get_tf()
print("test:", np.allclose(rfc, rfc2))
print(rfc.shape)
#
freq_rfc = np.zeros((4, len(freq_MHz)), dtype=rfc.dtype)
print(freq_rfc[1:].shape)
freq_rfc[1:] = rfc
freq_rfc[0] = freq_MHz

print(freq_rfc[:, 0])

plt.figure()
plt.title("RF Chain v1")
plt.plot(freq_rfc[0].real, np.abs(freq_rfc[1]), color="k", label="1")
plt.plot(freq_rfc[0].real, np.abs(freq_rfc[2]), color="y", label="2")
plt.plot(freq_rfc[0].real, np.abs(freq_rfc[3]), color="b", label="3")
plt.ylim([0, 90])
plt.grid()
plt.legend()


np.save("TF_RF_Chain", freq_rfc)


plt.show()
