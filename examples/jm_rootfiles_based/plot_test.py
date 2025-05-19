#!/usr/bin/env python3

from grand.dataio.root_files import get_file_event
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

version = 2
tag = "_jun12"

# Plot for new Leff to be consistent
Voc_1 = get_file_event("/home/grand/scripts/test_voltage_old_rfchain_newLeff_Voc.root")
Voc_o = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_Voc_oldLeff.root")
Voc_2 = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_Voc_newLeff.root")

V1 = get_file_event("/home/grand/scripts/test_voltage_old_rfchain_vgaP20dB_newLeff.root")
V2 = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_vgaP20dB_newLeff.root")
V3 = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_vgaP5dB_newLeff.root")
V4 = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_vga0dB_newLeff.root")
V5 = get_file_event(f"/home/grand/scripts/test_voltage_version{version}_rfchain_vgaM5dB_newLeff.root")

V6 = get_file_event(f"/home/grand/scripts/test_voltage_old_rfchain_oldLeff.root")
V7 = get_file_event(f"/home/grand/scripts/test_voltage_old_rfchain_oldLeff_manually.root")

E = get_file_event(f"/home/grand/data/test_efield.root")

#print(V_A.traces.shape, V_B.traces.shape)
idx_du=44

# Plot Efield
plt.figure()
plt.plot(E.traces[idx_du,0], color='k', label='Ex')
plt.plot(E.traces[idx_du,1], color='y', label='Ey')
plt.plot(E.traces[idx_du,2], color='b', label='Ez')
plt.ylabel(r"Efield ($\mu$V/m)")
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
plt.xlabel('Time bins')
plt.xlim(75,200)
plt.savefig("/home/grand/scripts/test_efield%s.png"%tag, bbox_inches="tight")

# Plot Voc
plt.figure()
plt.subplot(311)
plt.plot(Voc_1.traces[idx_du,0], color='k', ls="--", label='Old RFchain (grandlib)')
plt.plot(Voc_2.traces[idx_du,0], color='#808080', label=f'New RFchain (v{version})')
plt.ylabel(r"V$_{\rm oc} (\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
plt.title(r'V$_{\rm oc}$')
plt.title('X-port', fontsize=13)
# ------
plt.subplot(312)
plt.plot(Voc_1.traces[idx_du,1], color='y', ls="--", label='Old RFchain (grandlib)')
plt.plot(Voc_2.traces[idx_du,1], color='#808000', label=f'New RFchain (v{version})')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.grid(ls='--', alpha=0.3)
plt.title('Y-port', fontsize=13)
# ------
plt.subplot(313)
plt.plot(Voc_1.traces[idx_du,2], color='b', ls="--", label='Old RFchain (grandlib)')
plt.plot(Voc_2.traces[idx_du,2], color='#00FFFF', label=f'New RFchain (v{version})')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.grid(ls='--', alpha=0.3)
plt.xlabel('Time bins')
plt.title('Z-port', fontsize=13)
plt.savefig(f"/home/grand/scripts/test_voltage_version{version}_Voc{tag}.png", bbox_inches="tight")
plt.show()

# Plot voltage (rf chain without galactic noise)
plt.figure()
plt.subplot(311)
plt.plot(V1.traces[idx_du,0], color='k', ls="--", label='Old RFchain (grandlib, 20dB)')
plt.plot(V2.traces[idx_du,0], color='#808080', label=f'New RFchain (v{version}, 20dB)')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
plt.xlim(100,400)
plt.title('X-port', fontsize=13)
# ------
plt.subplot(312)
plt.plot(V1.traces[idx_du,1], color='y', ls="--", label='Old RFchain (grandlib)')
plt.plot(V2.traces[idx_du,1], color='#808000', label=f'New RFchain (v{version})')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.grid(ls='--', alpha=0.3)
plt.xlim(100,400)
plt.title('Y-port', fontsize=13)
# ------
plt.subplot(313)
plt.plot(V1.traces[idx_du,2], color='b', ls="--", label='Old RFchain (grandlib)')
plt.plot(V2.traces[idx_du,2], color='#00FFFF', label=f'New RFchain (v{version})')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.grid(ls='--', alpha=0.3)
plt.xlabel('Time bins')
plt.xlim(100,400)
plt.title('Z-port', fontsize=13)
plt.savefig(f"/home/grand/scripts/test_voltage_version{version}_rfchain%s{tag}.png", bbox_inches="tight")
plt.show()

# Plot voltage for different VGA gain.
plt.figure()
plt.subplot(311)
plt.plot(V2.traces[idx_du,0], color="#580F41", label='VGA gain: 20 dB')
plt.plot(V3.traces[idx_du,0], color="#E50000", label='VGA gain: 5 dB')
plt.plot(V4.traces[idx_du,0], color="#008080", label='VGA gain: 0 dB')
plt.plot(V5.traces[idx_du,0], color="#FBDD7E", label='VGA gain: -5 dB')
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
plt.xlim(100,400)
plt.title('X-port', fontsize=13)
# ------
plt.subplot(312)
plt.plot(V2.traces[idx_du,1], color="#580F41", label='VGA gain: 20 dB')
plt.plot(V3.traces[idx_du,1], color="#E50000", label='VGA gain: 5 dB')
plt.plot(V4.traces[idx_du,1], color="#008080", label='VGA gain: 0 dB')
plt.plot(V5.traces[idx_du,1], color="#FBDD7E", label='VGA gain: -5 dB')
plt.ylabel(r"Voltage ($\mu$V)")
plt.xticks(visible=False)
plt.grid(ls='--', alpha=0.3)
plt.xlim(100,400)
plt.title('Y-port', fontsize=13)
# ------
plt.subplot(313)
plt.plot(V2.traces[idx_du,2], color="#580F41", label='VGA gain: 20 dB')
plt.plot(V3.traces[idx_du,2], color="#E50000", label='VGA gain: 5 dB')
plt.plot(V4.traces[idx_du,2], color="#008080", label='VGA gain: 0 dB')
plt.plot(V5.traces[idx_du,2], color="#FBDD7E", label='VGA gain: -5 dB')
plt.grid(ls='--', alpha=0.3)
plt.xlabel('Time bins')
plt.xlim(100,400)
plt.title('Z-port', fontsize=13)
plt.savefig(f"/home/grand/scripts/test_voltage_version{version}_rfchain_different_gain{tag}.png", bbox_inches="tight")
plt.show()

# Plot voltage (rf chain without galactic noise)
plt.figure()
plt.subplot(311)
plt.plot(V6.traces[idx_du,0], color='k', ls="--", label='grandlib')
plt.plot(V7.traces[idx_du,0], color='#808080', label=f'manually')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.grid(ls="--", alpha=0.4)
plt.xlim(100,400)
plt.title('X-port', fontsize=13)
# ------
plt.subplot(312)
plt.plot(V6.traces[idx_du,1], color='y', ls="--", label='grandlib')
plt.plot(V7.traces[idx_du,1], color='#808000', label=f'manually')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.grid(ls='--', alpha=0.3)
plt.xlim(100,400)
plt.title('Y-port', fontsize=13)
# ------
plt.subplot(313)
plt.plot(V6.traces[idx_du,2], color='b', ls="--", label='grandlib')
plt.plot(V7.traces[idx_du,2], color='#00FFFF', label=f'manually')
plt.legend(loc='upper right')
plt.grid(ls='--', alpha=0.3)
plt.xlabel('Time bins')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xlim(100,400)
plt.title('Z-port', fontsize=13)
plt.savefig(f"/home/grand/scripts/test_voltage_grandlib_vs_manually{tag}.png", bbox_inches="tight")
plt.show()

# Plot voltage (rf chain without galactic noise)
plt.figure()
plt.subplot(311)
plt.plot(Voc_o.traces[idx_du,0], color='k', ls="--", label='Prev. Leff')
plt.plot(Voc_2.traces[idx_du,0], color='#808080', label=f'New Leff')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.xticks(visible=False)
plt.grid(ls="--", alpha=0.4)
plt.xlim(0,400)
plt.title('X-port', fontsize=13)
# ------
plt.subplot(312)
plt.plot(Voc_o.traces[idx_du,1], color='y', ls="--", label='Prev. Leff')
plt.plot(Voc_2.traces[idx_du,1], color='#808000', label=f'New Leff')
plt.ylabel(r"Voltage ($\mu$V)", fontsize=14)
plt.xticks(visible=False)
plt.legend(loc='upper right')
plt.xticks(visible=False)
plt.grid(ls='--', alpha=0.3)
plt.xlim(0,400)
plt.title('Y-port', fontsize=13)
# ------
plt.subplot(313)
plt.plot(Voc_o.traces[idx_du,2], color='b', ls="--", label='Prev. Leff')
plt.plot(Voc_2.traces[idx_du,2], color='#00FFFF', label=f'New Leff')
plt.legend(loc='upper right')
plt.grid(ls='--', alpha=0.3)
plt.xlabel('Time bins')
plt.ylabel(r"V$_{\rm oc} (\mu$V)", fontsize=14)
plt.title('Z-port', fontsize=13)
plt.xlim(0,400)
plt.savefig(f"/home/grand/scripts/previous_vs_new_Leff_voltage_version{version}{tag}.png", bbox_inches="tight")
plt.show()

