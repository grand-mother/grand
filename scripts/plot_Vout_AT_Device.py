#!/usr/bin/env python3

"""
Plot Voltage output and ratios at individual RF chain elements.
October 2024 SN
"""
# To Run:
#   python3 plot_Vout_AT_Device.py lna
#   options: [Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA Vout_tot] for Voltage at Device
#   options: [Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc] for Voltage ratios

import numpy as np
import h5py
import scipy.fft as sf
import grand.sim.detector.rf_chain as grfc
from grand import grand_add_path_data
import grand.manage_log as mlg

import matplotlib.pyplot as plt
params = {
    "legend.fontsize": 10,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.figsize": (8, 14),
    "axes.grid": True,
}
plt.rcParams.update(params)
l_col = ['k', 'y', 'b']
freq_MHz = np.arange(30, 251, 1)

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

def plot(args="Vin_balun1", savefig=False, **kwargs):
    
    if args=='Vin_balun1':

        print("Input Voltage at first Balun before Matching Network")
        rfchain= grfc.RFChain_in_Balun1()
        rfchain.compute_for_freqs(freq_MHz)    
        
        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Input Voltage (Real) at first Balun (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Input Voltage (Imag) at first Balun (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Input Voltage (abs) at first Balun (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Input_Voltage_Balun1.png", bbox_inches='tight')
        #plt.show()
        
    
    if args=='Vout_balun1':

        print("Output Voltage at first Balun before Matching Network")
        rfchain= grfc.RFChain_Balun1()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at first Balun (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at first Balun (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at first Balun (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_Balun1.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vout_match_net':

        print("Output Voltage at Matching Network")
        rfchain= grfc.RFChain_Match_net()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at Matching Network (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at Matching Network (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at Matching Network (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_match_net.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vout_lna':

        print("Output Voltage at LNA")
        rfchain= grfc.RFChainNut()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at LNA (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at LNA (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at LNA (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_lna.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vout_cable_connector':

        print("Output Voltage at Cable + Connector")
        rfchain= grfc.RFChain_Cable_Connectors()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at Cable + Connector (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at Cable + Connector (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at Cable + Connector (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_cable_connector.png", bbox_inches='tight')
        #plt.show()
        
    if args=='Vout_VGA':

        print("Output Voltage at VGA + Filters")
        rfchain= grfc.RFChain_VGA()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at VGA + Filters (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at VGA + Filters (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at VGA + Filters (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_vga.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vout_tot':

        print("Output Voltage at ADC")
        rfchain= grfc.RFChain()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.subplot(3, 1, 1)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.real(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("real(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Real) at ADC (V$_{oc}$=1)")
        plt.subplot(3, 1, 2)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.imag(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("imag(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (Imag) at ADC (V$_{oc}$=1)")
        plt.subplot(3, 1, 3)
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Output Voltage (abs) at ADC (V$_{oc}$=1)")
        if savefig:
            plt.savefig("Output_Voltage_ADC.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vratio_Balun1':

        print("Voltage ratio at Balun1")
        rfchain= grfc.RFChain_in_Balun1()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Balun1()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at Balun1")
        if savefig:
            plt.savefig("Voltage_ratio_Balun1.png", bbox_inches='tight')
        #plt.show()
     
    if args=='Vratio_match_net':

        print("Voltage ratio at Matching network")
        rfchain= grfc.RFChain_Balun1()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Match_net()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at Matching network")
        if savefig:
            plt.savefig("Voltage_ratio_match_net.png", bbox_inches='tight')
        #plt.show()
        
    if args=='Vratio_lna':

        print("Voltage ratio at LNA")
        rfchain= grfc.RFChain_Match_net()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChainNut()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at LNA")
        if savefig:
            plt.savefig("Voltage_ratio_lna.png", bbox_inches='tight')
        #plt.show()
        
        
    if args=='Vratio_cable_connector':

        print("Voltage ratio at Cable+Connector")
        rfchain= grfc.RFChainNut()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Cable_Connectors()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at Cable+Connector")
        if savefig:
            plt.savefig("Voltage_ratio_cable_connector.png", bbox_inches='tight')
        #plt.show()
        
    if args=='Vratio_vga':

        print("Voltage ratio at VGA+Filters")
        rfchain= grfc.RFChain_Cable_Connectors()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_VGA()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at VGA+Filters")
        if savefig:
            plt.savefig("Voltage_ratio_vga.png", bbox_inches='tight')
        #plt.show()
    
    if args=='Vratio_adc':

        print("Voltage ratio at ADC")
        rfchain= grfc.RFChain_VGA()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain()
        rfchain1.compute_for_freqs(freq_MHz)
        
        plt.figure(figsize=(7, 5))
        for port in range(3):
            plt.plot(rfchain.freqs_mhz, np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[port])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)")
        plt.ylabel("abs(FFT(V$_{out}$))/abs(FFT(V$_{in}$))", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.title("Voltage ratio at ADC")
        if savefig:
            plt.savefig("Voltage_ratio_adc.png", bbox_inches='tight')
        #plt.show()

    ##########################################################################################      

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parser to select which quantity to plot. \
        To Run: python3 plot_Vout_AT_Device.py <plot_option>. \
        <plot_option>: Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA, Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc \
        example: python3 plot_Vout_AT_Device.py Vout_lna --savefig"
        )
    parser.add_argument(
        "plot_option",
        help="what do you want to plot? example: Vout_lna.",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=False,
        help="don't add Voc.",
    )

    args = parser.parse_args()

    options_list = ["Vin_balun1", "Vout_balun1", "Vout_match_net", "Vout_lna", "Vout_cable_connector", "Vout_VGA", "Vout_tot", "Vratio_Balun1", "Vratio_match_net", "Vratio_lna", "Vratio_cable_connector", "Vratio_vga", "Vratio_adc"]

    if args.plot_option in options_list:
        plot(args.plot_option, savefig=args.savefig)
    else:
        raise Exception("Please provide a proper option for plotting noise. Options: Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna,  Vout_cable_connector, Vout_VGA Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc")