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
#   python3 plot_noise.py --lst 10 --savefig
#   python3 plot_noise.py --lst 18 --savefig --du_type GP300_nec

def plot(savefig=False, du_type='GP300_nec', **kwargs):
    du_type = kwargs.get('du_type', 'GP300_nec')  # Default value is 'GP300_nec'  
    if 'lst' in kwargs.keys():
        lst = kwargs['lst']
    else:
        raise Exception("Provide LST info like plot('galactic_noise', lst=..)")

    lst = int(lst)

    if du_type == 'GP300':
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_hfss.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_hfss.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_hfss.npy")
        gala_file3 = grand_add_path_data("noise/30_250galactic.mat")
        gala_show = h5py.File(gala_file3, "r")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1)) #micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1) 
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1)) #watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1)) # dBm per MHz
        gala_freq = gala_show["freq_all"]
    
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
        plt.savefig("galactic_noise.png", bbox_inches="tight")
    #plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot function with command line arguments")
    parser.add_argument('--savefig', action='store_true', help="Flag to save the figure")
    parser.add_argument('--du_type', type=str, default='GP300_nec', help="Type of du")
    parser.add_argument('--lst', type=int, required=True, help="LST info")

    args = parser.parse_args()

    # Convert args to dictionary
    kwargs = vars(args)

    # Call the plot function with provided arguments
    plot(**kwargs)

if __name__ == "__main__":
    main()

    
    
#if __name__ == "__main__":
#    import argparse
#
#    parser = argparse.ArgumentParser(
#        description="Parser to select which noise quantity to plot. \
#        To Run: ./plot_noise.py \
#        Add --lst <int> for galactic noise. i.e ./plot_noise.py --lst 18 --savefig."
#        )
#    parser.add_argument(
#        "--lst",
#        default=18.0,
#        type=float,
#        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h.",
#        )
#    parser.add_argument(
#        "--savefig",
#        action="store_true",
#        default=False,
#        help="don't add galactic noise.",
#    )
#
#    args = parser.parse_args()
#
#    plot(lst=args.lst, savefig=args.savefig)
#
