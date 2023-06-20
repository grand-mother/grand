#!/usr/bin/env python3

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

def plot(savefig=False, **kwargs):

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parser to select which noise quantity to plot. \
        To Run: ./plot_noise.py <plot_option>. \
        <plot_option>: galactic, vswr, lna, vga, cable, rf_chain. \
        Add --lst <int> for galactic noise. i.e ./plot_noise.py galactic --lst 18."
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


    plot(lst=args.lst, savefig=args.savefig)

