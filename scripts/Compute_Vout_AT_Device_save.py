#!/usr/bin/env python3

"""
Compute Voltage output and ratios at individual RF chain elements.
October 2024 SN
"""
# To Run:
#   python3 Compute_Vout_AT_Device_save.py lna
#   options: [Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA Vout_tot] for Voltage at Device
#   options: [Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc] for Voltage ratios

import numpy as np
import h5py
import scipy.fft as sf
import grand.sim.detector.rf_chain as grfc
from grand import grand_add_path_data
import grand.manage_log as mlg

freq_MHz = np.arange(30, 251, 1)

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

def save_data(args="Vin_balun1", savedata=False, **kwargs):
    
    if args=='Vin_balun1':

        print("Input Voltage at first Balun before Matching Network")
        rfchain= grfc.RFChain_in_Balun1()
        rfchain.compute_for_freqs(freq_MHz) 
        
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Input_Voltage_Balun1', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")
    
    if args=='Vout_balun1':

        print("Output Voltage at first Balun before Matching Network")
        rfchain= grfc.RFChain_Balun1()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_balun1', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")        
    
    if args=='Vout_match_net':

        print("Output Voltage at Matching Network")
        rfchain= grfc.RFChain_Match_net()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_match_net', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")
    
    if args=='Vout_lna':

        print("Output Voltage at LNA")
        rfchain= grfc.RFChainNut()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_lna', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")   
    
    if args=='Vout_cable_connector':

        print("Output Voltage at Cable + Connector")
        rfchain= grfc.RFChain_Cable_Connectors()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_cable_connector', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")
        
    if args=='Vout_VGA':

        print("Output Voltage at VGA + Filters")
        rfchain= grfc.RFChain_VGA()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_vga', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")
    
    if args=='Vout_tot':

        print("Output Voltage at ADC")
        rfchain= grfc.RFChain()
        rfchain.compute_for_freqs(freq_MHz)
        fr = rfchain.freqs_mhz
        V0 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0]
        V1 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1]
        V2 = rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2]
        print("Frequency[MHz], Vout_portX, Vout_portY, Vout_portZ")
        for a, b, c, d in zip(fr, V0, V1, V2):
            b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
            c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
            d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
            print(f"{a:.2f} {b_str} {c_str} {d_str}")
            #print(f"{a} {b} {c} {d}")
        data = np.column_stack((fr, V0, V1, V2))
        if savedata:
            with open('Output_Voltage_adc', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vout_port0 Vout_port1 Vout_port2\n')
                for a, b, c, d in zip(fr, V0, V1, V2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    b_str = f"{b.real:.2f}{'+' if b.imag >= 0 else '-'}{abs(b.imag):.2f}j"
                    c_str = f"{c.real:.2f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.2f}j"
                    d_str = f"{d.real:.2f}{'+' if d.imag >= 0 else '-'}{abs(d.imag):.2f}j"
                    f.write(f"{a:.2f} {b_str} {c_str} {d_str}\n")
            print("data file created")
           
    
    if args=='Vratio_Balun1':

        print("Voltage ratio at Balun1")
        rfchain= grfc.RFChain_in_Balun1()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Balun1()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_Balun1', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")
     
    if args=='Vratio_match_net':

        print("Voltage ratio at Matching network")
        rfchain= grfc.RFChain_Balun1()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Match_net()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_match_net', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")
        
    if args=='Vratio_lna':

        print("Voltage ratio at LNA")
        rfchain= grfc.RFChain_Match_net()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChainNut()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_lna', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")
        
        
    if args=='Vratio_cable_connector':

        print("Voltage ratio at Cable+Connector")
        rfchain= grfc.RFChainNut()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_Cable_Connectors()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_cable_connector', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")
        
    if args=='Vratio_vga':

        print("Voltage ratio at VGA+Filters")
        rfchain= grfc.RFChain_Cable_Connectors()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain_VGA()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_vga', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")
            
    if args=='Vratio_adc':

        print("Voltage ratio at ADC")
        rfchain= grfc.RFChain_VGA()
        rfchain.compute_for_freqs(freq_MHz)
        rfchain1= grfc.RFChain()
        rfchain1.compute_for_freqs(freq_MHz)
        ratio_0 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[0])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[0])
        ratio_1 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[1])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[1])
        ratio_2 = np.abs(rfchain1.vout_f(np.ones((3,rfchain1.nb_freqs)))[2])/np.abs(rfchain.vout_f(np.ones((3,rfchain.nb_freqs)))[2])
        fr = rfchain.freqs_mhz
        print("Frequency[MHz], Vratio_portX, Vratio_portY, Vratio_portZ")
        for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
            print(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}")        
        data = np.column_stack((rfchain.freqs_mhz,ratio_0,ratio_1,ratio_2))
        if savedata:
            with open('Voltage_ratio_adc', 'w') as f:
                # Write header
                f.write('frequency[MHz] Vratio_port0 Vratio_port1 Vratio_port2\n')
                for a, b, c, d in zip(fr, ratio_0, ratio_1, ratio_2):
                    # Format complex numbers to ensure negative imaginary parts are printed correctly
                    f.write(f"{a:.2f} {b:.3f} {c:.3f} {d:.3f}\n")
            print("data file created")

    ##########################################################################################      

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parser to select which quantity to Compute. \
        To Run: python3 Compute_Vout_AT_Device_save.py <save_data_option>. \
        <save_data_option>: Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna, Vout_cable_connector, Vout_VGA, Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc \
        example: python3 Compute_Vout_AT_Device_save.py Vout_lna --savedata"
        )
    parser.add_argument(
        "save_data_option",
        help="what do you want to print/save? example: Vout_lna.",
    )
    parser.add_argument(
        "--savedata",
        action="store_true",
        default=False,
        help="Create a txt file to store data.",
    )

    args = parser.parse_args()

    options_list = ["Vin_balun1", "Vout_balun1", "Vout_match_net", "Vout_lna", "Vout_cable_connector", "Vout_VGA", "Vout_tot", "Vratio_Balun1", "Vratio_match_net", "Vratio_lna", "Vratio_cable_connector", "Vratio_vga", "Vratio_adc"]

    if args.save_data_option in options_list:
        save_data(args.save_data_option, savedata=args.savedata)
    else:
        raise Exception("Please provide a proper option for print and save output voltage. Options: Vin_balun1, Vout_balun1, Vout_match_net, Vout_lna,  Vout_cable_connector, Vout_VGA Vout_tot, Vratio_Balun1, Vratio_match_net, Vratio_lna, Vratio_cable_connector, Vratio_vga, Vratio_adc")