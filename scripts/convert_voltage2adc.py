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

Optionally (--t1_trigger, off by default) the offline DAQ-style T1 trigger of
`grand.sim.detector.trigger` is applied to every DU, and `trigger_flag` is set
per DU (1 = passed T1, 0 = not).  Its parameters can be changed with
--t1_param KEY=VALUE; the defaults are those of `scripts/T1_trigger_offline.py`
and are still to be confirmed by the trigger group.  Without --t1_trigger the
output is unchanged.
'''

###-###-###-###-###-###-###- IMPORTS -###-###-###-###-###-###-###

import glob
import os
import time
import argparse
import logging
import psutil
import numpy as np
# import matplotlib.pyplot as plt

from grand import ADC, manage_log
import grand.dataio
from grand.sim.detector.trigger import DEFAULT_T1_CONFIG, t1_trigger_flags

logger = logging.getLogger(__name__)


###-###-###-###-###-###-###- FUNCTIONS -###-###-###-###-###-###-###

def noise_files(data_dir):
    r"""Returns the noise data files for `data_dir`, sorted.

    A directory is searched inside, with or without a trailing '/'.  Without
    the slash this used to glob the directory's *siblings*, find nothing, and
    fill every ADC sample with 8192 (issue #122).  Anything that is not a
    directory is kept as a path prefix, as before, e.g. '/data/GP80_2025'.

    Parameters
    ----------
    data_dir : str
        A directory of noise ROOT files, or a path prefix.

    Returns
    -------
    list of str
        The matching files, sorted to remove glob's ordering.

    Raises
    ------
    FileNotFoundError
        If nothing matches, rather than simulating from no noise at all.
    """
    pattern = (os.path.join(data_dir, '*.root') if os.path.isdir(data_dir)
               else data_dir + '*.root')
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(f'No noise files match {pattern}')
    return found


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
    data_files = noise_files(data_dir)

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

    # print(f"mem1: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    for i, data_file in enumerate(data_files):

        if i < n_traces % n_files:
            n_entries_sel = n_traces // n_files + 1
        else:
            n_entries_sel = n_traces // n_files

        if n_entries_sel == 0: continue

        # print(f"mem2: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        df = grand.dataio.DataFile(data_file)
        tadc = df.tadc #rt.TADC(data_file)

        # tadc = grand.dataio.TADC(data_file)
        # print(f"mem3: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Check that data traces contain requested number of samples
        # tadc.get_entry(0)
        tadc._tree.GetBranch("adc_samples_count_ch").GetEntry(0)
        n_samples_data = tadc.adc_samples_count_ch[0][1] #TODO: tempfix

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

        entries_sel = rng.integers(0,high=n_entries_tot,size=n_entries_sel)
        logger.debug(f'Selected {n_entries_sel} random traces from {data_file}')

        trace_branch = tadc._tree.GetBranch("trace_ch")
        du_id_branch = tadc._tree.GetBranch("du_id")

        for entry in entries_sel:
            # print(f"mem4: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            # res = tadc.get_entry(entry)
            res = trace_branch.GetEntry(int(entry))
            # print(f"mem5: {process.memory_info().rss / 1024 ** 2:.2f} MB", res)
            trace = np.array(tadc.trace_ch)[0,:,:n_samples]

            #-- START OF ADDITION TO EXTEND NOISE TRACES --#
            # This can be removed once we take data that is 2048 samples instead of 1024

            if extend_noise_trace:
                rms  = np.sqrt( np.mean( trace**2,axis=1 ) )
                mean = np.mean( trace,axis=1 )
                extend_condition = False

                # Only extend with data from same DU
                du_id_branch.GetEntry(int(entry))
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
        # print(f"mem8: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    return noise_trace


def t1_config_from_params(params):
    r"""The T1 trigger parameters, from ``KEY=VALUE`` strings.

    Parameters
    ----------
    params : list of str or None
        Overrides of :data:`grand.sim.detector.trigger.DEFAULT_T1_CONFIG`,
        e.g. ``['th1=120', 'nc_max=10']``.  The values are integers.

    Returns
    -------
    dict
        The full set of trigger parameters.

    Raises
    ------
    ValueError
        For a string that is not ``KEY=VALUE`` with an integer value, or an
        unknown key.
    """
    config = dict(DEFAULT_T1_CONFIG)
    for param in params or []:
        key, sep, value = param.partition('=')
        key = key.strip()
        if not sep or key not in DEFAULT_T1_CONFIG:
            raise ValueError(f'Bad --t1_param {param!r}: expected KEY=VALUE with KEY in {sorted(DEFAULT_T1_CONFIG)}')
        try:
            config[key] = int(value)
        except ValueError:
            raise ValueError(f'Bad --t1_param {param!r}: the value must be an integer') from None
    return config


def apply_t1_trigger(tadc, adc_trace, t1_config):
    r"""Sets ``tadc.trigger_flag`` from the T1 trigger on every DU.

    Parameters
    ----------
    tadc : grand.dataio.TADC
        The tree whose current entry is being filled.
    adc_trace : numpy.ndarray
        The ADC traces, with shape (N_du, 3, N_samples).
    t1_config : dict
        The trigger parameters, see :func:`t1_config_from_params`.

    Returns
    -------
    numpy.ndarray of numpy.ushort
        The flags written, 1 for a DU that passed T1 and 0 otherwise.
    """
    flags = t1_trigger_flags(adc_trace, t1_config)
    tadc.trigger_flag = flags
    return flags


def manage_args(argv=None):
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

    parser.add_argument('--t1_trigger',
                        action='store_true',
                        help='Apply the offline DAQ-style T1 trigger to every DU and set trigger_flag '
                             '(1 = passed, 0 = not). Default off, leaving the output unchanged.')

    parser.add_argument('--t1_param',
                        action='append',
                        metavar='KEY=VALUE',
                        default=None,
                        help='Override a T1 trigger parameter (repeatable), e.g. --t1_param th1=120. '
                             f'Defaults: {DEFAULT_T1_CONFIG}. Only used with --t1_trigger.')

    return parser.parse_args(argv)


###-###-###-###-###-###-###- MAIN SCRIPT -###-###-###-###-###-###-###

if __name__ == '__main__':
    logger = manage_log.get_logger_for_script(__file__)
    pid = os.getpid()
    process = psutil.Process(pid)    
    

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
    t1_config = t1_config_from_params(args.t1_param) if args.t1_trigger else None

    manage_log.create_output_for_logger(args.verbose,log_stdout=True)
    logger.info( manage_log.string_begin_script() )
    if t1_config is not None:
        logger.info(f'Applying the T1 trigger to every DU, with parameters {t1_config}')
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    #-#-#- Load TVoltage -#-#-#
    df       = grand.dataio.DataDirectory(f_input_dir)
    tvoltage = df.tvoltage
    entries  = tvoltage.get_number_of_entries()
    trun = df.trun

    # Loop through the voltage files
    for f_input_file in df.ftvoltages[0].flist:

        df_input_file = grand.dataio.DataFile(f_input_file)
        tvoltage = df_input_file.tvoltage
        entries = tvoltage.get_number_of_entries()

        logger.info(f'Converting {entries} voltage traces from {f_input_file} to ADC traces')
        print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        if args.out_file is None:
            # Replace only first occurrences
            f_output = "adc".join(f_input_file.split("voltage", 1))
            f_output = "L1".join(f_output.split("L0", 1))

        #-#-#- Prepare TADC -#-#-#
        if os.path.exists(f_output):
            logger.info(f"Overwriting {f_output}") # remove existing file if it already exists
            os.remove(f_output)
            time.sleep(1)
        tadc = grand.dataio.TADC(f_output)

        #-#-#- Initiate ADC object and RNG -#-#-#
        adc = ADC()

        rng = np.random.default_rng(args.seed)
        if noise_dir is not None:
            logger.info(f'Set RNG seed to {args.seed}')
            logger.info(f'Adding random measured noise traces from data files in {noise_dir}')


        #-#-#- Perform the conversion for all entries in TVoltage file -#-#-#
        for entry in range(entries):
            logger.info(f'Converting voltage to ADC for entry {entry+1}/{entries}')
            # print(f"Entry Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
            res = tvoltage.get_entry(entry)
            # print(f"Memory 1: {process.memory_info().rss / 1024 ** 2:.2f} MB", res)
            # voltage_trace = np.array(tvoltage.trace, copy=True)
            voltage_trace = np.array(tvoltage.trace)
            # print(f"Memory 2: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            event_number = tvoltage.event_number
            run_number = tvoltage.run_number

            # A shower that hit no antenna (issue #91): nothing to digitise, but
            # the event is still written, with du_count 0 and empty traces.
            if voltage_trace.size == 0:
                logger.warning(f'Event {event_number} of run {run_number} has no antenna (du_count 0): '
                               'no ADC trace to compute; it is written with du_count 0.')
                tadc.copy_contents(tvoltage)
                tadc.trace_ch = np.zeros((0, 3, 0), dtype=np.int16)
                tadc.trigger_position = np.zeros(0, dtype=np.ushort)
                tadc.fill()
                continue

            trun.get_run(run_number)
            # print(f"Memory 2_1: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            event_dus_indices = tvoltage.get_dus_indices_in_run(trun)
            dt_ns = np.asarray(trun.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns.
            f_samp_mhz = 1e3/dt_ns                                 # MHz
            input_sampling_rate_mhz = f_samp_mhz[0]                # and here we asume all sampling rates are the same!. In any case, we are asuming all the ADCs are the same...
            # print(f"Memory 3: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            #-#-#- Downsample if needed -#-#-# (this could be added to the "process" method to hide it from the public, and add input_sampling_rate as input to process.
            #plt.plot(voltage_trace[1][1],label="in")
            if( input_sampling_rate_mhz != adc.sampling_rate):
               voltage_trace=adc.downsample(voltage_trace,input_sampling_rate_mhz)
               #plt.plot(voltage_trace[1][1],label="downsampled")
            #-#-#- Get noise trace if requested -#-#-#
            # print(f"Memory before get noise: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            if noise_dir is not None:
                noise_trace = get_noise_trace(noise_dir,
                                              voltage_trace.shape[0],
                                              n_samples=voltage_trace.shape[2],
                                              rng=rng)

            # print(f"Memory after get noise: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            #-#-#- Convert voltage trace to adc trace -#-#-#
            adc_trace = adc.process(voltage_trace,
                                    noise_trace=noise_trace)
            # print(f"Memory after adding noise: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            #plt.plot(adc_trace[1][1],label="adc")
            #plt.show()
            #-#-#- Save adc trace to TADC file -#-#-#
            tadc.copy_contents(tvoltage)
            # print(f"Memory after copy: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            entries_adc = tadc.get_number_of_entries()
            tadc.trace_ch = adc_trace
            # print(f"Memory after tracemod: {process.memory_info().rss / 1024 ** 2:.2f} MB")


            #modify the trigger position if needed. TODO: the T1 trigger (--t1_trigger) only sets trigger_flag; the trigger position still comes from the simulation
            if(input_sampling_rate_mhz != adc.sampling_rate):
              originalsampling=input_sampling_rate_mhz
              newsampling=adc.sampling_rate
              ratio=originalsampling/newsampling
            else:
              ratio=1.0

            tadc.trigger_position=np.ushort(np.asarray(tvoltage.trigger_position)/ratio)

            #-#-#- Optional T1 trigger, per DU -#-#-#
            if t1_config is not None:
                flags = apply_t1_trigger(tadc, adc_trace, t1_config)
                logger.info(f'T1 trigger: {int(np.count_nonzero(flags))}/{len(flags)} DUs passed')
            # print(f"Memory after trig mod: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            tadc.fill()
            # print(f"Memory after fill: {process.memory_info().rss / 1024 ** 2:.2f} MB")
            logger.debug(f'ADC trace for (run,event) = {tvoltage.run_number, tvoltage.event_number} written to TADC')


        tadc.analysis_level = tadc.analysis_level+1
        tadc.write()
        logger.info(f'Succesfully saved TADC to {f_output}')

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    logger.info( manage_log.string_end_script() )
