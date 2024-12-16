#!/usr/bin/env python3

"""
Plot qunatities related to galactic noise and RF chain.
"""
import numpy as np
import h5py
import argparse
from grand import grand_add_path_data

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

# To Run:
#   python3 plot_noise.py
#   python3 plot_noise.py --lst 10 --savefig (Default value is 'GP300')
#   python3 plot_noise.py --lst 18 --savefig --du_type GP300_nec

def plot(savefig=False, du_type='GP300', **kwargs):  
    if 'lst' in kwargs.keys():
        lst = kwargs['lst']
    else:
        raise Exception("Provide LST info like plot('galactic_noise', lst=..)")

    lst = int(lst)
    
    if du_type == 'GP300':
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
        #plt.show()
    
    elif du_type == 'GP300_nec':
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_nec.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_nec.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_nec.npy")
        gala_file3 = grand_add_path_data("noise/30_250galactic.mat")
        gala_show = h5py.File(gala_file3, "r")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1)) #micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1) 
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1)) #watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1)) # dBm per MHz
        gala_freq = gala_show["freq_all"]
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/MHz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        # -----
        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_watt[:, l_g, lst])
        # SL: gala_power_dbm = 1e6 * np.sqrt(2 * 100 * pow(10, gala_power_dbm/10) * 1e-3)
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(Watt/MHz)", fontsize=15)
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
            plt.savefig("galactic_noise_nec.png", bbox_inches="tight")
        #plt.show()
        
    elif du_type == 'GP300_mat':
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_mat.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_mat.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_mat.npy")
        gala_file3 = grand_add_path_data("noise/30_250galactic.mat")
        gala_show = h5py.File(gala_file3, "r")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1)) #micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1) 
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1)) #watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1)) # dBm per MHz
        gala_freq = gala_show["freq_all"]
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/MHz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        # -----
        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_watt[:, l_g, lst])
        # SL: gala_power_dbm = 1e6 * np.sqrt(2 * 100 * pow(10, gala_power_dbm/10) * 1e-3)
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(Watt/MHz)", fontsize=15)
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
            plt.savefig("galactic_noise_mat.png", bbox_inches="tight")
        #plt.show()
    else:
        raise ValueError(f"Unknown du_type: {du_type}")

        
def main():
    parser = argparse.ArgumentParser(description="Plot function with command line arguments")
    parser.add_argument('--savefig', action='store_true', help="Flag to save the figure")
    parser.add_argument('--du_type', type=str, default='GP300', help="Type of du")
    parser.add_argument('--lst', type=float, default=18, help="LST info (defaults to 18)")

    args = parser.parse_args()
    # Call the plot function with provided arguments
    plot(savefig=args.savefig, lst=args.lst, du_type=args.du_type)

if __name__ == "__main__":
    main()
