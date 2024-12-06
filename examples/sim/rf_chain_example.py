#!/usr/bin/env python
"""
Plot qunatities related to galactic noise and RF chain.
"""
import numpy as np
import h5py
import scipy.fft as sf

import grand.sim.detector.rf_chain as grfc
from grand import grand_add_path_data
import grand.manage_log as mlg

import matplotlib.pyplot as plt
params = {
    "legend.fontsize": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    #"figure.figsize": (10, 8),
    "axes.grid": False,
}
plt.rcParams.update(params)

l_col = ['k', 'y', 'b']
freq_MHz = np.arange(30, 251, 1)

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

# To Run:
#   python3 plot_noise.py galactic_noise
#   options: [galactic, vswr, lna, vga, cable, rf_chain]

def plot(args="galactic", savefig=False, **kwargs):

    if args=="galactic":
        if 'lst' in kwargs.keys():
            lst = kwargs['lst']
        else:
            raise Exception("Provide LST info like plot('galactic_noise', lst=..)")

        lst = int(lst)

        gala_file = grand_add_path_data("noise/30_250galactic.mat")
        gala_show = h5py.File(gala_file, "r")
        gala_psd_dbm   = np.transpose(gala_show["psd_narrow_huatu"])
        # SL, dbm per MHz, P=mean(V*V)/imp with imp=100 ohms
        gala_power_dbm = np.transpose(gala_show["p_narrow_huatu"])
        # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V)) 
        gala_voltage = np.transpose(gala_show["v_amplitude"])  
        # gala_power_mag = np.transpose(gala_show["p_narrow"])
        gala_freq = gala_show["freq_all"]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(gala_freq, gala_psd_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        # -----
        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_dbm[:, l_g, lst])
        # SL: gala_power_dbm = 1e6 * np.sqrt(2 * 100 * pow(10, gala_power_dbm/10) * 1e-3)
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        # -----
        plt.subplot(1, 3, 3)
        for l_g in range(3):
            plt.plot(gala_freq, gala_voltage[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"Voltage($\rm\mu$V)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.grid(ls='--', alpha=0.3)
        plt.subplots_adjust(top=0.85)
        if savefig:
            plt.savefig("galactic_noise.png", bbox_inches="tight")
        plt.show()

    if args=='lna':

        print("Parameters of LNA")

        lna = grfc.LowNoiseAmplifier()
        lna.compute_for_freqs(freq_MHz)

        """
        plot of LNA S11
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"LNA S$_{11}$ [Interpolated]")
        plt.plot(lna.freqs_mhz, np.real(lna.s11[0]), "k")
        plt.plot(lna.freqs_mhz, np.real(lna.s11[1]), "y")
        plt.plot(lna.freqs_mhz, np.real(lna.s11[2]), "b")
        plt.ylabel(r"Real(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(lna.freqs_mhz, np.imag(lna.s11[0]), "k")
        plt.plot(lna.freqs_mhz, np.imag(lna.s11[1]), "y")
        plt.plot(lna.freqs_mhz, np.imag(lna.s11[2]), "b")
        plt.ylabel(r"Imag(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(lna.freqs_mhz, np.abs(lna.s11[0]), "k")
        plt.plot(lna.freqs_mhz, np.abs(lna.s11[1]), "y")
        plt.plot(lna.freqs_mhz, np.abs(lna.s11[2]), "b")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of LNA S21
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"LNA S$_{21}$ [Interpolated]")
        plt.plot(lna.freqs_mhz, np.real(lna.s21[0]), "k")
        plt.plot(lna.freqs_mhz, np.real(lna.s21[1]), "y")
        plt.plot(lna.freqs_mhz, np.real(lna.s21[2]), "b")
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.ylabel(r"Real(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(lna.freqs_mhz, np.imag(lna.s21[0]), "k")
        plt.plot(lna.freqs_mhz, np.imag(lna.s21[1]), "y")
        plt.plot(lna.freqs_mhz, np.imag(lna.s21[2]), "b")
        plt.ylabel(r"Imag(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(lna.freqs_mhz, np.abs(lna.s21[0]), "k")
        plt.plot(lna.freqs_mhz, np.abs(lna.s21[1]), "y")
        plt.plot(lna.freqs_mhz, np.abs(lna.s21[2]), "b")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of LNA S12
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"LNA S$_{12}$ [Interpolated]")
        plt.plot(lna.freqs_mhz, np.real(lna.s12[0]), "k")
        plt.plot(lna.freqs_mhz, np.real(lna.s12[1]), "y")
        plt.plot(lna.freqs_mhz, np.real(lna.s12[2]), "b")
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.ylabel(r"Real(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(lna.freqs_mhz, np.imag(lna.s12[0]), "k")
        plt.plot(lna.freqs_mhz, np.imag(lna.s12[1]), "y")
        plt.plot(lna.freqs_mhz, np.imag(lna.s12[2]), "b")
        plt.ylabel(r"Imag(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(lna.freqs_mhz, np.abs(lna.s12[0]), "k")
        plt.plot(lna.freqs_mhz, np.abs(lna.s12[1]), "y")
        plt.plot(lna.freqs_mhz, np.abs(lna.s12[2]), "b")
        plt.ylabel(r"abs(S$_{12}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of LNA S22
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"LNA S$_{22}$ [Interpolated]")
        plt.plot(lna.freqs_mhz, np.real(lna.s22[0]), "k")
        plt.plot(lna.freqs_mhz, np.real(lna.s22[1]), "y")
        plt.plot(lna.freqs_mhz, np.real(lna.s22[2]), "b")
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.ylabel(r"Real(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(lna.freqs_mhz, np.imag(lna.s22[0]), "k")
        plt.plot(lna.freqs_mhz, np.imag(lna.s22[1]), "y")
        plt.plot(lna.freqs_mhz, np.imag(lna.s22[2]), "b")
        plt.ylabel(r"Imag(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(lna.freqs_mhz, np.abs(lna.s22[0]), "k")
        plt.plot(lna.freqs_mhz, np.abs(lna.s22[1]), "y")
        plt.plot(lna.freqs_mhz, np.abs(lna.s22[2]), "b")
        plt.ylabel(r"abs(S$_{22}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of LNA S-parameters in dB
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(lna.sparams[port][:, 0]/1e6, lna.sparams[port][:, 1], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{11}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.title("LNA S-parameters [dB]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(lna.sparams[port][:, 0]/1e6, lna.sparams[port][:, 3], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{21}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(lna.sparams[port][:, 0]/1e6, lna.sparams[port][:, 5], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{12}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(lna.sparams[port][:, 0]/1e6, lna.sparams[port][:, 7], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"S$_{22}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        """
        plot of all LNA S-parameters
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(lna.freqs_mhz, np.abs(lna.s11[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{11}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title("LNA S-parameters [Interpolated]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(lna.freqs_mhz, np.abs(lna.s21[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{21}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(lna.freqs_mhz, np.abs(lna.s12[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{12}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(lna.freqs_mhz, np.abs(lna.s22[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(S$_{22}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    if args=="balun_after_lna":

        print("Parameters of Balun after LNA")

        obj = grfc.BalunAfterLNA()
        obj.compute_for_freqs(freq_MHz)

        """
        plot S11 of Balun after LNA
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun after LNA [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[2]), "b")
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.ylabel(r"Real(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[2]), "b")
        plt.ylabel(r"Imag(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[2]), "b")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S21 of Balun after LNA
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun after LNA [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[2]), "b")
        plt.ylabel(r"Real(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[2]), "b")
        plt.ylabel(r"Imag(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[2]), "b")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S12 of Balun after LNA
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun after LNA [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[0]), "k", label=r"0")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[1]), "y", label=r"1")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[2]), "b", label=r"2")
        plt.ylabel(r"Real(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[2]), "b")
        plt.ylabel(r"Imag(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[2]), "b")
        plt.ylabel(r"abs(S$_{12}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S22 of Balun after LNA
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun after LNA [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[2]), "b")
        plt.ylabel(r"Real(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[2]), "b")
        plt.ylabel(r"Imag(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[2]), "b")
        plt.ylabel(r"abs(S$_{22}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        # Balun after LNA S-parameters
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s11[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{11}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title("Balun after LNA [Interpolated]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s21[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{21}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s12[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{12}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s22[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(S$_{22}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    if args=='cable':
        print("Parameters of cable and connector")

        cable  = grfc.Cable()
        cable.compute_for_freqs(freq_MHz)

        """
        plot of Cable S11
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"Cable S$_{11}$ [Interpolated]")
        plt.plot(cable.freqs_mhz, np.real(cable.s11[0]), "k")
        plt.plot(cable.freqs_mhz, np.real(cable.s11[1]), "y")
        plt.plot(cable.freqs_mhz, np.real(cable.s11[2]), "b")
        plt.ylabel(r"Real(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(cable.freqs_mhz, np.imag(cable.s11[0]), "k")
        plt.plot(cable.freqs_mhz, np.imag(cable.s11[1]), "y")
        plt.plot(cable.freqs_mhz, np.imag(cable.s11[2]), "b")
        plt.ylabel(r"Imag(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(cable.freqs_mhz, np.abs(cable.s11[0]), "k")
        plt.plot(cable.freqs_mhz, np.abs(cable.s11[1]), "y")
        plt.plot(cable.freqs_mhz, np.abs(cable.s11[2]), "b")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of Cable S21
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"Cable S$_{21}$ [Interpolated]")
        plt.plot(cable.freqs_mhz, np.real(cable.s21[0]), "k")
        plt.plot(cable.freqs_mhz, np.real(cable.s21[1]), "y")
        plt.plot(cable.freqs_mhz, np.real(cable.s21[2]), "b")
        plt.ylabel(r"Real(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(cable.freqs_mhz, np.imag(cable.s21[0]), "k")
        plt.plot(cable.freqs_mhz, np.imag(cable.s21[1]), "y")
        plt.plot(cable.freqs_mhz, np.imag(cable.s21[2]), "b")
        plt.ylabel(r"Imag(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(cable.freqs_mhz, np.abs(cable.s21[0]), "k")
        plt.plot(cable.freqs_mhz, np.abs(cable.s21[1]), "y")
        plt.plot(cable.freqs_mhz, np.abs(cable.s21[2]), "b")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of Cable S12
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"Cable S$_{12}$ [Interpolated]")
        plt.plot(cable.freqs_mhz, np.real(cable.s12[0]), "k")
        plt.plot(cable.freqs_mhz, np.real(cable.s12[1]), "y")
        plt.plot(cable.freqs_mhz, np.real(cable.s12[2]), "b")
        plt.ylabel(r"Real(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(cable.freqs_mhz, np.imag(cable.s12[0]), "k")
        plt.plot(cable.freqs_mhz, np.imag(cable.s12[1]), "y")
        plt.plot(cable.freqs_mhz, np.imag(cable.s12[2]), "b")
        plt.ylabel(r"Imag(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(cable.freqs_mhz, np.abs(cable.s12[0]), "k")
        plt.plot(cable.freqs_mhz, np.abs(cable.s12[1]), "y")
        plt.plot(cable.freqs_mhz, np.abs(cable.s12[2]), "b")
        plt.ylabel(r"abs(S$_{12}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of Cable S22
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"Cable S$_{22}$ [Interpolated]")
        plt.plot(cable.freqs_mhz, np.real(cable.s22[0]), "k")
        plt.plot(cable.freqs_mhz, np.real(cable.s22[1]), "y")
        plt.plot(cable.freqs_mhz, np.real(cable.s22[2]), "b")
        plt.ylabel(r"Real(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(cable.freqs_mhz, np.imag(cable.s22[0]), "k")
        plt.plot(cable.freqs_mhz, np.imag(cable.s22[1]), "y")
        plt.plot(cable.freqs_mhz, np.imag(cable.s22[2]), "b")
        plt.ylabel(r"Imag(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(cable.freqs_mhz, np.abs(cable.s22[0]), "k")
        plt.plot(cable.freqs_mhz, np.abs(cable.s22[1]), "y")
        plt.plot(cable.freqs_mhz, np.abs(cable.s22[2]), "b")
        plt.ylabel(r"abs(S$_{22}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        # Cable S-parameters in dB
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(cable.sparams[:, 0]/1e6, cable.sparams[:, 1], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{11}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.title("Cable S-parameters [dB]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(cable.sparams[:, 0]/1e6, cable.sparams[:, 3], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{21}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(cable.sparams[:, 0]/1e6, cable.sparams[:, 5], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{12}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(cable.sparams[:, 0]/1e6, cable.sparams[:, 7], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"S$_{22}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Cable S-parameters
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(cable.freqs_mhz, np.abs(cable.s11[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{11}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title("Cable S-parameters [Interpolated]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(cable.freqs_mhz, np.abs(cable.s21[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{21}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(cable.freqs_mhz, np.abs(cable.s12[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{12}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(cable.freqs_mhz, np.abs(cable.s22[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(S$_{22}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    if args=='vga':
        print("Parameters of VGA and filter")

        vga = grfc.VGAFilter()
        vga.compute_for_freqs(freq_MHz)

        """
        plot of VGA+Filter S11
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"VGA+Filter S$_{11}$ [Interpolated]")
        plt.plot(vga.freqs_mhz, np.real(vga.s11[0]), "k")
        plt.plot(vga.freqs_mhz, np.real(vga.s11[1]), "y")
        plt.plot(vga.freqs_mhz, np.real(vga.s11[2]), "b")
        plt.ylabel(r"Real(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(vga.freqs_mhz, np.imag(vga.s11[0]), "k")
        plt.plot(vga.freqs_mhz, np.imag(vga.s11[1]), "y")
        plt.plot(vga.freqs_mhz, np.imag(vga.s11[2]), "b")
        plt.ylabel(r"Imag(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(vga.freqs_mhz, np.abs(vga.s11[0]), "k")
        plt.plot(vga.freqs_mhz, np.abs(vga.s11[1]), "y")
        plt.plot(vga.freqs_mhz, np.abs(vga.s11[2]), "b")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of VGA+Filter S21
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"VGA+Filter S$_{21}$ [Interpolated]")
        plt.plot(vga.freqs_mhz, np.real(vga.s21[0]), "k")
        plt.plot(vga.freqs_mhz, np.real(vga.s21[1]), "y")
        plt.plot(vga.freqs_mhz, np.real(vga.s21[2]), "b")
        plt.ylabel(r"Real(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(vga.freqs_mhz, np.imag(vga.s21[0]), "k")
        plt.plot(vga.freqs_mhz, np.imag(vga.s21[1]), "y")
        plt.plot(vga.freqs_mhz, np.imag(vga.s21[2]), "b")
        plt.ylabel(r"Imag(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(vga.freqs_mhz, np.abs(vga.s21[0]), "k")
        plt.plot(vga.freqs_mhz, np.abs(vga.s21[1]), "y")
        plt.plot(vga.freqs_mhz, np.abs(vga.s21[2]), "b")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of VGA+Filter S12
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"VGA+Filter S$_{12}$ [Interpolated]")
        plt.plot(vga.freqs_mhz, np.real(vga.s12[0]), "k")
        plt.plot(vga.freqs_mhz, np.real(vga.s12[1]), "y")
        plt.plot(vga.freqs_mhz, np.real(vga.s12[2]), "b")
        plt.ylabel(r"Real(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(vga.freqs_mhz, np.imag(vga.s12[0]), "k")
        plt.plot(vga.freqs_mhz, np.imag(vga.s12[1]), "y")
        plt.plot(vga.freqs_mhz, np.imag(vga.s12[2]), "b")
        plt.ylabel(r"Imag(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(vga.freqs_mhz, np.abs(vga.s12[0]), "k")
        plt.plot(vga.freqs_mhz, np.abs(vga.s12[1]), "y")
        plt.plot(vga.freqs_mhz, np.abs(vga.s12[2]), "b")
        plt.ylabel(r"abs(S$_{12}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot of VGA+Filter S22
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title(r"VGA+Filter S$_{22}$ [Interpolated]")
        plt.plot(vga.freqs_mhz, np.real(vga.s22[0]), "k")
        plt.plot(vga.freqs_mhz, np.real(vga.s22[1]), "y")
        plt.plot(vga.freqs_mhz, np.real(vga.s22[2]), "b")
        plt.ylabel(r"Real(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(vga.freqs_mhz, np.imag(vga.s22[0]), "k")
        plt.plot(vga.freqs_mhz, np.imag(vga.s22[1]), "y")
        plt.plot(vga.freqs_mhz, np.imag(vga.s22[2]), "b")
        plt.ylabel(r"Imag(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(vga.freqs_mhz, np.abs(vga.s22[0]), "k")
        plt.plot(vga.freqs_mhz, np.abs(vga.s22[1]), "y")
        plt.plot(vga.freqs_mhz, np.abs(vga.s22[2]), "b")
        plt.ylabel(r"abs(S$_{22}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        # VGA+Filter S-parameters in dB
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(vga.sparams[:, 0]/1e6, vga.sparams[:, 1], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{11}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.title("VGA+Filter S-parameters [dB]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(vga.sparams[:, 0]/1e6, vga.sparams[:, 3], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{21}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(vga.sparams[:, 0]/1e6, vga.sparams[:, 5], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"S$_{12}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(vga.sparams[:, 0]/1e6, vga.sparams[:, 7], l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"S$_{22}$ [dB]", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # VGA+Filter S-parameters
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(vga.freqs_mhz, np.abs(vga.s11[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{11}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title("VGA+Filter S-parameters [Interpolated]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(vga.freqs_mhz, np.abs(vga.s21[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{21}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(vga.freqs_mhz, np.abs(vga.s12[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{12}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(vga.freqs_mhz, np.abs(vga.s22[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(S$_{22}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    if args=='balun_before_adc':
        print("Parameters of Balun before ADC")

        obj = grfc.BalunBeforeADC()
        obj.compute_for_freqs(freq_MHz)
        """
        plot S11 of Balun before ADC
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun before ADC [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s11[2]), "b")
        plt.ylabel(r"Real(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s11[2]), "b")
        plt.ylabel(r"Imag(S$_{11}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s11[2]), "b")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S21 of Balun before ADC
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun before ADC [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s21[2]), "b")
        plt.ylabel(r"Real(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s21[2]), "b")
        plt.ylabel(r"Imag(S$_{21}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s21[2]), "b")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S12 of Balun before ADC
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun before ADC [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s12[2]), "b")
        plt.ylabel(r"Real(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s12[2]), "b")
        plt.ylabel(r"Imag(S$_{12}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s12[2]), "b")
        plt.ylabel(r"abs(S$_{12}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        """
        plot S22 of Balun before ADC
        """
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        plt.title("Balun before ADC [Interpolated]")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.real(obj.s22[2]), "b")
        plt.ylabel(r"Real(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        # ----
        plt.subplot(3, 1, 2)
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.imag(obj.s22[2]), "b")
        plt.ylabel(r"Imag(S$_{22}$)")
        plt.xticks(visible=False)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[0]), "k")
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[1]), "y")
        plt.plot(obj.freqs_mhz, np.abs(obj.s22[2]), "b")
        plt.ylabel(r"abs(S$_{22}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)

        # Balun before ADC S-parameters
        plt.figure(figsize=(8, 12))
        plt.subplot(4, 1, 1)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s11[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{11}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title("Balun before ADC [Interpolated]")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 2)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s21[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{21}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(4, 1, 3)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s12[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"abs(S$_{12}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # ----
        plt.subplot(4, 1, 4)
        for port in range(3):
            plt.plot(obj.freqs_mhz, np.abs(obj.s22[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(S$_{22}$)", fontsize=15)
        plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    if args=='rf_chain':

        print("Parameters of total RF Chain")

        gain = 20      # VGA options: [-5, 0, 5, 20]
        name_dict = {-5:'M5', 0:'0', 5:'P5', 20:'P20'}
        rfchain= grfc.RFChain(vga_gain=gain)
        rfchain.compute_for_freqs(freq_MHz)

        # Antenna Impedance. Zant
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.real(rfchain.Z_ant[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"Real(Z$_{\rm ant}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.title("Antenna Impedance")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.imag(rfchain.Z_ant[port]), l_col[port])
        plt.xticks(visible=False)
        plt.ylabel(r"Imag(Z$_{\rm ant}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.abs(rfchain.Z_ant[port]), l_col[port])
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(Z$_{\rm ant}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Input Impedance. Zin
        plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.real(rfchain.Z_in[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xticks(visible=False)
        plt.ylabel(r"Real(Z$_{\rm in}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.title(r"Z$_{\rm in}$")
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.imag(rfchain.Z_in[port]), l_col[port])
        plt.xticks(visible=False)
        plt.ylabel(r"Imag(Z$_{\rm in}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        # ----
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.lna.freqs_mhz, np.abs(rfchain.Z_in[port]), l_col[port])
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"abs(Z$_{\rm in}$)", fontsize=15)
        #plt.xlim(30, 250)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Load Impedance. Zload
        plt.figure(figsize=(10, 8))
        plt.plot(rfchain.lna.freqs_mhz, np.real(rfchain.Z_load[0]), 'b')
        plt.plot(rfchain.lna.freqs_mhz, np.imag(rfchain.Z_load[0]), 'orange')
        plt.legend(["Real", "Imag"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"Z$_{\rm load} (\Omega)$", fontsize=15)
        plt.title("Load Impedance")
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.figure()
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
            #print('port:', port)
            #print(list(rfchain.freqs_mhz))
            #print(list(np.abs(rfchain.vout(np.ones((3,rfchain.nb_freqs)))[port])))
            #print('')
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel(r"Transfer Function: abs(V$_{\rm out}$/V$_{\rm oc}$)", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title(f"VGA gain: {gain} dB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parser to select which noise quantity to plot. \
        To Run: ./plot_noise.py <plot_option>. \
        <plot_option>: galactic, vswr, lna, vga, cable, rf_chain. \
        Add --lst <int> for galactic noise. i.e ./plot_noise.py galactic --lst 18."
        )
    parser.add_argument(
        "plot_option",
        default="galactic",
        help="Option to select which noise quantity to plot.",
        )
    parser.add_argument(
        "--lst",
        default=18.0,
        type=float,
        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h.",
        )

    parser.add_argument(
        "--savefig",
        action="store_true",
        default=False,
        help="don't add galactic noise.",
    )

    args = parser.parse_args()

    options_list = ["lna", "balun_after_lna", "vga", "cable", "balun_before_adc", "rf_chain"]

    if args.plot_option=="galactic":
        plot(args.plot_option, lst=args.lst, savefig=args.savefig)
    elif args.plot_option in options_list:
        plot(args.plot_option, savefig=args.savefig)
    else:
        raise Exception("Please provide a proper option for plotting noise. Options: galactic, vswr, lna, vga, cable, rf_chain.")
    plt.show()
