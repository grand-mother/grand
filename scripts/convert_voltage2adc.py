#! /usr/bin/env python3
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

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import time
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

#from grand import ADC, manage_log
from grand.sim.detector.adc import ADC
import grand.manage_log
import grand.dataio.root_trees as rt

logger = logging.getLogger(__name__)


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
    description : Number of data files to consider for the selection. Default selects all files.

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

    # Select n_files random data files from directory
    data_files = sorted( glob.glob(data_dir+'*.root') ) # sort to get rid of additional randomness of glob

    if n_files is None:
        n_files = len(data_files)

    assert n_files <= len(data_files), f'There are {len(data_files)} in {data_dir} - requested {n_files}'
    idx_files = rng.choice( range( len(data_files) ), n_files, replace=False )
    data_files = [data_files[i] for i in idx_files]

    logger.info(f'Fetching {n_traces} random noise traces of 3 x {n_samples} samples from {n_files} data files in {data_dir}')

    # Reduce files to open if n_traces < n_files
    quotient  = n_files // n_traces
    remainder = n_files % n_traces

    if quotient == 0:
        data_files = data_files[:remainder]
        logger.debug(f'Only need to open {remainder} < {n_traces} data files')

    # Get noise traces from data files
    noise_trace = np.empty( (n_traces,3,n_samples),dtype=int )
    trace_idx = 0

    for i, data_file in enumerate(data_files):
        df = rt.DataFile(data_file)
        tadc = df.tadc #rt.TADC(data_file)

        # Check that data traces contain requested number of samples
        tadc.get_entry(0)
        n_samples_data = tadc.adc_samples_count_ch[0][1]*2 #TODO: tempfix

        if n_samples_data == n_samples/2:
            extend_noise_trace = True
            logger.warning(f'Two random data traces of {n_samples_data} samples will be concatenated to obtain noise traces of {n_samples} samples')
            logger.warning(f'This is SLOW! Suggest to merge traces first. See e.g. `/pbs/home/p/pcorrea/grand/dc2/scripts/merge_noise_trace.py`')
        else:
            extend_noise_trace = False
            assert n_samples_data >= n_samples, f'Data trace contains less samples than requested: {n_samples_data} < {n_samples}'

        # Select random entries from TADC
        # NOTE: assumed that each entry corresponds to a single DU with ADC channels (0,1,2)=(X,Y,Z)
        n_entries_tot = tadc.get_number_of_entries()
        
        if i < n_traces % n_files:
            n_entries_sel = n_traces // n_files + 1
        else:
            n_entries_sel = n_traces // n_files
        
        entries_sel = rng.integers(0,high=n_entries_tot,size=n_entries_sel)
        logger.debug(f'Selected {n_entries_sel} random traces from {data_file}')
        
        for entry in entries_sel:
            tadc.get_entry(entry)
            trace = np.array(tadc.trace_ch)[0,:,:n_samples]

            #-- START OF ADDITION TO EXTEND NOISE TRACES --#
            # This can be removed once we take data that is 2048 samples instead of 1024

            if extend_noise_trace:
                rms  = np.sqrt( np.mean( trace**2,axis=1 ) )
                mean = np.mean( trace,axis=1 )
                extend_condition = False

                # Only extend with data from same DU
                du_id = tadc.du_id[0] 

                # Only extend the original trace with a new trace
                # if the relative RMS difference between them is <10%
                # and if the baseline difference between them is <10%
                # in both X and Y channels
                entry_ext = entry
                while not extend_condition:
                    entry_ext = (entry_ext + 1) % n_entries_tot
                    tadc.get_entry(entry_ext)

                    if tadc.du_id[0] != du_id:
                        continue

                    trace_ext = np.array(tadc.trace_ch)[0,:,:n_samples]
                    rms_ext   = np.sqrt( np.mean( trace_ext**2,axis=1 ) )
                    mean_ext  = np.mean( trace_ext,axis=1 )

                    rms_diff  = np.abs(rms_ext-rms)/rms
                    mean_diff = np.abs(mean_ext-mean)/mean

                    if np.all( rms_diff[:2] < 0.05) and np.all( mean_diff[:2] < 0.1):
                        extend_condition = True

                trace = np.append(trace,trace_ext,axis=1)

            #-- END OF ADDITION TO EXTEND NOISE TRACES --#

            noise_trace[trace_idx] = trace
            trace_idx += 1
        df.close()
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
    parser.add_argument(
                        "--target_sampling_rate_mhz",
                        type=float,
                        default=0,
                        help="Target sampling rate of the data in Mhz (not implemented, currently hard coded to 500Mhz)",
    )       
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
    f_input_dir   = args.in_file
    f_output  = args.out_file
    noise_dir = args.noise_dir

    f_input_file=glob.glob(f_input_dir+"/voltage_*_L0_*.root")[0]

    if f_output == None:
        f_output = f_input_file.replace('voltage','adc')
        f_output = f_output.replace('L0','L1')
    if noise_dir == None:
        noise_trace = None

    manage_log.create_output_for_logger(args.verbose,log_stdout=True)
    logger.info( manage_log.string_begin_script() )
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    logger.info(f'Converting voltage traces from {f_input_file} to ADC traces')

    #-#-#- Load TVoltage -#-#-#
    df       = rt.DataDirectory(f_input_dir)
    tvoltage = df.tvoltage
    entries  = tvoltage.get_number_of_entries()
    trun = df.trun

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

        event_number = tvoltage.event_number
        run_number = tvoltage.run_number
        trun.get_run(run_number)
        event_dus_indices = tvoltage.get_dus_indices_in_run(trun)
        dt_ns = np.asarray(trun.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 
        f_samp_mhz = 1e3/dt_ns                                 # MHz  
        input_sampling_rate_mhz = f_samp_mhz[0]                # and here we asume all sampling rates are the same!. In any case, we are asuming all the ADCs are the same...        

        #-#-#- Downsample if needed -#-#-# (this could be added to the "process" method to hide it from the public, and add input_sampling_rate as input to process.
        #plt.plot(voltage_trace[1][1],label="in")
        if( input_sampling_rate_mhz != adc.sampling_rate):         
           voltage_trace=adc.downsample(voltage_trace,input_sampling_rate_mhz)
           #plt.plot(voltage_trace[1][1],label="downsampled")
        #-#-#- Get noise trace if requested -#-#-#
        if noise_dir is not None:
            noise_trace = get_noise_trace(noise_dir,
                                          voltage_trace.shape[0],
                                          n_samples=voltage_trace.shape[2],
                                          rng=rng)
        #-#-#- Convert voltage trace to adc trace -#-#-#
        adc_trace = adc.process(voltage_trace,
                                noise_trace=noise_trace)

        #plt.plot(adc_trace[1][1],label="adc")
        #plt.show()
        #-#-#- Save adc trace to TADC file -#-#-#
        tadc.copy_contents(tvoltage)
        entries  = tadc.get_number_of_entries()
        tadc.trace_ch = adc_trace
        

        #modify the trigger position if needed. TODO: This will have at some point to be replaced by a real trigger algorithm
        if(input_sampling_rate_mhz != adc.sampling_rate):
          originalsampling=input_sampling_rate_mhz
          newsampling=adc.sampling_rate
          ratio=originalsampling/newsampling
        else:
          ratio=1.0   
        
        tadc.trigger_position=np.ushort(np.asarray(tvoltage.trigger_position)/ratio)

        tadc.fill()
        logger.debug(f'ADC trace for (run,event) = {tvoltage.run_number, tvoltage.event_number} written to TADC')
    

    tadc.analysis_level = tadc.analysis_level+1
    tadc.write()
    logger.info(f'Succesfully saved TADC to {f_output}')
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    logger.info( manage_log.string_end_script() )
