#! /usr/bin/env python3

import glob
import os
import time
import argparse
import logging

import numpy as np

from grand import ADC, manage_log
import grand.dataio.root_trees as rt

logger = logging.getLogger(__name__)

'''
Script to convert voltage traces to ADC traces.
Its main purpose is to create simulation files that resemble measured data.

This script performs the following tasks:
- reads a voltage simulation file, containing a TVoltage tree with voltage traces processed through the RF chain
- converts the analog voltage traces to digital ADC traces
- includes an option to add measured noise to the ADC traces
- save the (noisy) ADC traces in a TADC tree

This essentially acts as a follow-up script to `./convert_efield2voltage.py`.
NOTE: if noise is added from measured data, the input voltage trace should NOT include simulated galactic noise.

TO RUN:
    python convert_voltage2adc.py <voltage.root> -o <adc.root> --add_noise_from <noise_dir> -s <seed>
'''


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def get_noise_trace(data_dir,
                    n_traces,
                    n_files=None,
                    n_samples=2048,
                    rng=np.random.default_rng()):
    '''
    Selects random ADC noise traces from a directory containing files of measured data.

    Arguments
    ---------
    `data_dir`
    type        : str
    description : Path to directory where data files are stored in GrandRoot format.

    `n_traces`
    type        : int
    description : Number of noise traces to select, each with shape (3,n_samples).

    `n_files` (optional)
    type        : int
    description : Number of data files to consider for the selection. Default takes all files.

    `n_samples` (optional)
    type        : int
    description : Number of samples required from the measured data trace.

    `rng` (optional)
    type        : np.random.Generator
    description : Random number generator. Default has an unspecified seed. NOTE: default or seed=None makes the selection irreproducible. 
                                
    Returns
    -------
    `noise_trace`
    type        : np.ndarray[int]
    units       : ADC counts (least significant bits)
    description : The selected array of noise traces, with shape (N_du,3,N_samples).
    '''
    
    # This part might be patched with rt.DataDirectory() once compatible
    data_files = glob.glob(data_dir+'*.root')
    if n_files == None:
        n_files    = len(data_files)

    logger.info(f'Fetching {n_traces} random noise traces of 3 x {n_samples} samples from {n_files} data files in {data_dir}')

    quotient  = n_traces // n_files
    remainder = n_traces % n_files

    # Reduce files to open if n_traces < n_files
    if quotient == 0:
        data_files = data_files[:remainder]
        logger.debug(f'Only need to open {remainder} < {n_traces} data files')

    noise_trace = np.empty( (n_traces,3,n_samples),dtype=int )
    trace_idx = 0

    for i, data_file in enumerate(data_files):
        df = rt.DataFile(data_file)
        tadc = df.tadc #rt.TADC(data_file)

        # Check that data traces contain requested number of samples
        tadc.get_entry(0)
        n_samples_data = tadc.adc_samples_count_ch[0][1]
        assert n_samples_data >= n_samples, f'Data trace contains less samples than requested: {n_samples_data} < {n_samples}'

        # Select random entries from TADC
        # NOTE: assumed that each entry corresponds to a single DU with ADC channels (0,1,2)=(X,Y,Z)
        n_entries_tot = tadc.get_number_of_entries()
        
        if i < remainder:
            n_entries_sel = quotient + 1
        else:
            n_entries_sel = quotient
        
        entries_sel = rng.integers(0,high=n_entries_tot,size=n_entries_sel)
        logger.debug(f'Selected {n_entries_sel} random traces from {data_file}')

        for entry in entries_sel:
            tadc.get_entry(entry)

            # NOTE: important that a possible floating channel is NOT included here.
            # This would be done when creating dedicated noise files to add to simulations.
            # For code-testing with GP13 data, you have to remove channel 0 (= float channel)
            trace = np.array(tadc.trace_ch)[:,:,:n_samples]      
            #trace = np.array(tadc.trace_ch)[:,1:,:n_samples]        
            noise_trace[trace_idx] = trace
            trace_idx += 1

    return noise_trace


def manage_args():
    '''
    Manager for the argument parser of this script.
    '''

    parser = argparse.ArgumentParser(description="Conversion of voltage at ADC input to digitized ADC counts. Includes option to add measured noise.")

    parser.add_argument('in_file',
                        type=str,
                        help='Path to voltage input file in GrandRoot format (TVoltage).')
    
    parser.add_argument('-o',
                        '--out_file',
                        type=str,
                        default=None,
                        help='Path to utput file in GrandRoot format (TADC). If the file exists it is overwritten.')
    
    parser.add_argument('--add_noise_from',
                        dest='noise_dir',
                        type=str,
                        default=None,
                        help='Path to directory containing files with measured noise in GrandRoot format (TADC). Default adds no noise.')
    
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=None,
                        help='Fix the random seed for selection of measured noise traces. Must be positive integer. Default yields irreproducible RNG.')
    
    parser.add_argument('-v',
                        '--verbose',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info',
                        help='Logger verbosity.')

    return parser.parse_args()


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    logger = manage_log.get_logger_for_script(__file__)

    #-#-#- Get parser arguments -#-#-#
    args      = manage_args()
    f_input   = args.in_file
    f_output  = args.out_file
    noise_dir = args.noise_dir

    if f_output == None:
        f_output = f_input.replace('.root','_converted_to_ADC.root')
    if noise_dir == None:
        noise_trace = None

    manage_log.create_output_for_logger(args.verbose,log_stdout=True)
    logger.info( manage_log.string_begin_script() )
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    logger.info(f'Converting voltage traces from {f_input} to ADC traces')

    #-#-#- Load TVoltage -#-#-#
    df       = rt.DataFile(f_input)
    tvoltage = df.tvoltage
    entries  = tvoltage.get_number_of_entries()

    #-#-#- Prepare TADC -#-#-#
    if os.path.exists(f_output):
        logger.info(f"Overwriting {f_output}") # remove existing file if it already exists
        os.remove(f_output)
        time.sleep(1)
    tadc = rt.TADC(f_output)
    
    #-#-#- Initiate ADC object and RNG -#-#-#
    adc = ADC()
    rng = np.random.default_rng(args.seed)
    if noise_dir is not None:
        logger.info(f'Set RNG seed to {args.seed}')
        logger.info(f'Adding random measured noise traces from data files in {noise_dir}')
    

    #-#-#- Perform the conversion for all entries in TVoltage file -#-#-#
    for entry in range(entries):
        logger.info(f'Converting voltage to ADC for entry {entry+1}/{entries}')
        tvoltage.get_entry(entry)
        voltage_trace = np.array(tvoltage.trace)

        #-#-#- Get noise trace if requested -#-#-#
        if noise_dir is not None:
            noise_trace = get_noise_trace(noise_dir,
                                          voltage_trace.shape[0],
                                          n_samples=voltage_trace.shape[2],
                                          rng=rng)

        #-#-#- Convert voltage trace to adc trace -#-#-#
        adc_trace = adc.process(voltage_trace,
                                noise_trace=noise_trace)

        #-#-#- Save adc trace to TADC file -#-#-#
        tadc.copy_contents(tvoltage)
        tadc.trace_ch = adc_trace
        tadc.fill()
        tadc.write()
        logger.debug(f'ADC trace for (run,event) = {tvoltage.run_number, tvoltage.event_number} written to TADC')

    logger.info(f'Succesfully saved TADC to {f_output}')

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    logger.info( manage_log.string_end_script() )