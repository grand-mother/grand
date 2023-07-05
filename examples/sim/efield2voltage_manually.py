#!/usr/bin/env python3

"""
Example file to manually add noises and RF chain to voltage computation.

To Run:
    python example_e2v_manually.py ../data/test_efield.root -o jpt.root
"""
import numpy as np
from grand.sim.efield2voltage import Efield2Voltage
from grand.sim.noise.galaxy import galactic_noise
import grand.sim.detector.rf_chain as grfc

rng = np.random.default_rng(0)     

import grand.manage_log as mlg
# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

def noise(size):
    amp   = rng.normal(loc=0, scale=1, size=size)
    phase = 2 * np.pi * rng.random(size=size)
    v_complex = np.abs(amp) * np.exp(1j * phase)
    return v_complex

def transfer_func(size):
    return rng.random(size)

def style1(f_input, f_output):
    '''
    Example to show how to add or multiply to output voltage as many times as you want.
    '''
    logger.info("Computing voltage from the input electric field manually.")

    signal = Efield2Voltage(f_input, f_output, seed=0, padding_factor=1.2)
    nb_events = len(signal.events_list)  # total number of events

    # loop over each event.
    for evt_idx in range(nb_events):
        print(f"Working on event {evt_idx}")
        # computes voc and voc_f (Voc in time and frequency domain) for all DUs of evt_idx.
        signal.compute_voc_event(evt_idx) # event_idx=int. 

        # Add noise
        noise1 = noise(signal.voc_f.shape)    # shape = (nb_du, 3, nb_freq_trace)
        signal.add(noise1)

        # Multiply by some transfer functions
        lna_tf = transfer_func(len(signal.freqs_mhz))
        cable_tf = transfer_func(len(signal.freqs_mhz))
        vga_tf = transfer_func(len(signal.freqs_mhz))
        filter_tf = transfer_func(len(signal.freqs_mhz))
        signal.multiply(lna_tf)
        signal.multiply(cable_tf)
        signal.multiply(vga_tf)
        signal.multiply(filter_tf)
        # or simply
        #signal.multiply(lna_tf * cable_tf * vga_tf * filter_tf)

        # Add another noise at the end
        noise2 = noise(signal.voc_f.shape)
        signal.add(noise2)

        # compute the final voltage in time domain
        signal.final_voltage()

        # save the final voltage and other attributes as ROOT trees in the 'outfile' file.
        signal.save_voltage()

def style2(f_input, f_output):
    '''
    Example of manually reproducing voltage output after adding noise and rf chain.
    '''
    logger.info("Computing voltage from the input electric field manually.")

    # instantiate a class to compute voltage from efield provided in the infile file.
    signal = Efield2Voltage(f_input, f_output, seed=0, padding_factor=1)
    signal.compute_voc_event(0) # event_idx=int. Compute Voc.

    # add galactic noise after Voc is computed.
    fft_noise_gal_3d = galactic_noise(
        18,
        signal.fft_size,
        signal.freqs_mhz,
        signal.nb_du,
        seed=0
        )
    signal.add(fft_noise_gal_3d) # add galactic noise to the Voc.

    # add rf_chain
    rf_chain = grfc.RFChain()
    rf_chain.compute_for_freqs(signal.freqs_mhz)
    transfer_func = rf_chain.get_tf() # (3,nb_freq_trace)
    signal.multiply(transfer_func) # propagate signal through the RF chain.

    # compute the final voltage in time domain
    signal.final_voltage()

    # save the final voltage and other attributes as ROOT trees in the 'outfile' file.
    signal.save_voltage()

if __name__ == "__main__":
    import argparse
    import numpy as np
    import grand.manage_log as mlg

    # specific logger definition for script because __mane__ is "__main__" !
    logger = mlg.get_logger_for_script(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        help="Efield input data file in GRANDROOT format.",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default="../data/test_voltage_manually.root",
        help="output file in GRANDROOT format. If the file exists it is overwritten.",
        required=True,
        # PB with option ???
        # type=argparse.FileType("w"),
    )
    args = parser.parse_args()

    style1(args.file.name, args.out_file)
    #style2(args.file.name, args.out_file)

