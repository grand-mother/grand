#! /usr/bin/env python3
"""
Script to compute a "hardwarelike" efield from electric field.
Electric field traces are provided in a ROOT file.
   
usage: convert_efield2efield.py [-h] [--no_filter] [-o OUT_FILE] [-od OUT_DIRECTORY] [--verbose {debug,info,warning,error,critical}] [--seed SEED] [--add_noise_uVm ADD_NOISE_UVM] [--add_jitter_ns ADD_JITTER_NS]
                                [--calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA] [--target_duration_us TARGET_DURATION_US] [--target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ]
                                directory

Calculation of Hardware-like Efield input file.

positional arguments:
  directory             Simulation output data directory in GRANDROOT format.

optional arguments:
  -h, --help            show this help message and exit
  --no_filter           remove the filter on the GRAND bandwidth. (50-200Mhz, band-pass elliptic causal filter)
  -o OUT_FILE, --out_file OUT_FILE
                        output file in GRANDROOT format. If the file exists it is overwritten.
  -od OUT_DIRECTORY, --out_directory OUT_DIRECTORY
                        output directory in GRANDROOT format. If not given, is it the same as input directory
  --verbose {debug,info,warning,error,critical}
                        logger verbosity.
  --seed SEED           Fix the random seed to reproduce same galactic noise, must be positive integer
  --add_noise_uVm ADD_NOISE_UVM
                        level of gaussian noise (uv/m) to add to the trace before filtering
  --add_jitter_ns ADD_JITTER_NS
                        level of gaussian jitter (ns) to add to the trigger times
  --calibration_smearing_sigma CALIBRATION_SMEARING_SIGMA
                        Smear the stations amplitude calibrations with a gaussian centered in 1 and this input sigma
  --target_duration_us TARGET_DURATION_US
                        Adjust (and override) padding factor in order to get a signal of the given duration, in us
  --target_sampling_rate_mhz TARGET_SAMPLING_RATE_MHZ
                        Target sampling rate of the data in Mhz



March 2024, M Tueros. It was a shitty winter in Paris.
"""
def manage_args():
    parser = argparse.ArgumentParser(
        description="Calculation of Hardware-like Efield input file."
    )
    parser.add_argument(
        "directory",
        help="Simulation output data directory in GRANDROOT format.",
        # type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--no_filter",
        action="store_false",
        default=True,
        help="remove the filter on the GRAND bandwidth. (50-200Mhz, band-pass elliptic causal filter)",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default=None,
        help="output file in GRANDROOT format, relative to -od when it is a bare name. If the file "
             "exists it is replaced, as are the L1 run files.",
    )
    parser.add_argument(
        "-od",
        "--out_directory",
        default=None,
        help="output directory for the efield and run files. If not given, the input directory",
    )
    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logger verbosity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1, # -1 as a placeholder for None to maintain int type.
        help="Fix the random seed to reproduce same galactic noise, must be positive integer",
    )
    parser.add_argument(
        "--add_noise_uVm",
        type=float,
        default=0,
        help="level of gaussian noise (uv/m) to add to the trace before filtering",
    )
    parser.add_argument(
        "--add_jitter_ns",
        type=float,
        default=0,
        help="level of gaussian jitter (ns) to add to the trigger times",
    )
    parser.add_argument(
        "--calibration_smearing_sigma",
        type=float,
        default=0,
        help="Smear the stations amplitude calibrations with a gaussian centered in 1 and this input sigma",
    )    
    parser.add_argument(
        "--target_duration_us",
        type=float,
        default=0,
        help="Adjust (and override) padding factor in order to get a signal of the given duration, in us",
    )    
    parser.add_argument(
        "--target_sampling_rate_mhz",
        type=float,
        default=0,
        help="Target sampling rate of the data in Mhz",
    )      
    # retrieve argument
    return parser.parse_args()

def level1_name(level0_file, output_directory=None):
    r"""Returns the L1 counterpart of `level0_file`, in `output_directory` if given.

    Only the file name's ``_L0_`` changes: replacing "L0" in the whole path
    changed a folder's name instead when it held "L0".
    """
    name = Path(level0_file).name.replace("_L0_", "_L1_", 1)
    return str(Path(output_directory or Path(level0_file).parent) / name)


def get_fastest_size_fft(sig_size, f_samp_mhz, padding_factor=1):
    """
    :param sig_size:            length of time traces (samples)
    :param f_samp_mhz:          sampling frequency in MHz. ex: 2000 MHz for dt_ns=0.5
    :param padding_factor:      factor to stretch length of time traces with zeros
    :return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_factor >= 1    
    dt_s      = 1e-6 / f_samp_mhz
    fast_size = sf.next_fast_len(int(padding_factor * sig_size + 0.5))
    # ToDo: this function (or something higher) should properly handle different time bin for each trace
    fast_freqs_mhz = sf.rfftfreq(fast_size, dt_s[0]) * 1e-6
    #print(f"padding_factor {padding_factor} sig_size {sig_size} ({padding_factor * sig_size +0.5}) fast size {fast_size} fast_freqs_mhz size {len(fast_freqs_mhz)}")
    return fast_size, fast_freqs_mhz


# Function to depict magnitude 
# and phase plot
def mfreqz(b, a, Fs):
 
    # Compute frequency response of the
    # filter using signal.freqz function
    wz, hz = ss.freqz(b, a)
 
    # Calculate Magnitude from hz in dB
    Mag = 20*np.log10(abs(hz))
 
    # Calculate phase angle in degree from hz
    Phase = np.unwrap(np.arctan2(np.imag(hz), 
                                 np.real(hz)))*(180/np.pi)
 
    # Calculate frequency in Hz from wz
    Freq = wz*Fs/(2*np.pi)
 
    # Plot filter magnitude and phase responses using subplot.
    fig = plt.figure(figsize=(10, 6))
 
    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.axis([1, Fs/2, -100, 5])
    sub1.set_title('Magnitude Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=20)
    sub1.set_ylabel('Magnitude [dB]', fontsize=20)
    sub1.grid()
 
    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=20)
    sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
    sub2.set_title(r'Phase response', fontsize=20)
    sub2.grid()
 
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show()


# Define impz(b,a) to calculate impulse
# response and step response of a system
# input: b= an array containing numerator
# coefficients,a= an array containing
# denominator coefficients
def impz(b, a):
 
    # Define the impulse sequence of length 60
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
 
    # Compute the impulse response
    response = ss.lfilter(b, a, impulse)
 
    # Plot filter impulse and step response:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)
 
    plt.subplot(212)
    step = np.cumsum(response)
 
    # Compute step response of the system
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)
 
    fig.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    import argparse
    import os
    from typing import Union
    import numpy as np
    from pathlib import Path
    import grand.manage_log as mlg
    import grand.dataio
    import grand.geo.coordinates as coord
    from grand.basis.type_trace import ElectricField        
    import scipy.fft as sf
    import scipy.signal as ss
    import matplotlib.pyplot as plt  
    import glob  

    ################# Manage logger and input arguments ######################################
    PLOT=False #enable/disable plots when running.
    PlotAnt=3
    
    logger = mlg.get_logger_for_script(__name__)
    args = manage_args()
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    logger.info("Computing electric field from the input electric field.")

    # If no output directory given, define it as input directory
    if args.out_directory is None:
        args.out_directory = args.directory
    # A folder not yet made failed with FileNotFoundError (#182)
    os.makedirs(args.out_directory, exist_ok=True)

    seed = args.seed
    if seed < -1:
        raise SystemExit("GRANDlib: convert_efield2efield: --seed must be a non-negative integer, got %d" % seed)
    logger.info(f"seed used for random number generator is {seed}.")

    noise = args.add_noise_uVm
    if not 0 <= noise < float("inf"):    # a bare AssertionError before (#233); inf wrote NaN (#288)
        raise SystemExit("GRANDlib: convert_efield2efield: --add_noise_uVm must be finite and >= 0, got %s" % noise)
    if(noise>0):
      logger.info(f"We are going to apply gaussian noise of {noise} uV/m.")   
 
    jitter= args.add_jitter_ns
    if not 0 <= jitter < float("inf"):    # a bare AssertionError before (#233); inf wrote NaN (#288)
        raise SystemExit("GRANDlib: convert_efield2efield: --add_jitter_ns must be finite and >= 0, got %s" % jitter)
    if(jitter>0):
      logger.info(f"We are going to apply a gaussian time jitter of {jitter} ns")   
 
    calsigma=args.calibration_smearing_sigma
    if not 0 <= calsigma < float("inf"):    # a bare AssertionError before (#233); inf wrote NaN (#288)
        raise SystemExit("GRANDlib: convert_efield2efield: --calibration_smearing_sigma must be finite and >= 0, got %s" % calsigma)
    if(calsigma>0):
      logger.info(f"We are going to apply a gaussian calibration error of {calsigma} ")   
    
 
    padding_factor=1
    target_sampling_rate_mhz = args.target_sampling_rate_mhz   # if different from 0, will resample  
    # 0.001 or 0.5 (GHz typed for MHz) wrote traces of 0 to 4 samples (#288)
    if not (target_sampling_rate_mhz == 0 or 10 <= target_sampling_rate_mhz < float("inf")):
        raise SystemExit("GRANDlib: convert_efield2efield: --target_sampling_rate_mhz must be 0 (keep) or at least 10 MHz, got %s"
                         % target_sampling_rate_mhz)
    target_duration_us = args.target_duration_us       # if different from 0, will adjust padding factor to get a trace of this lenght in us        
    if not target_duration_us >= 0:
        raise SystemExit("GRANDlib: convert_efield2efield: --target_duration_us must be >= 0, got %s"
                         % target_duration_us)
    
    filter = args.no_filter
    f_output=args.out_file
    output_directory=args.out_directory
    #############################################################################################
    #############################################################################################
    def efield_output(f_input_file):
        r"""Returns where the L1 efield made from `f_input_file` goes."""
        f_output = args.out_file
        if f_output is None:
            # Replace only first occurrences
            f_output = "L1".join(f_input_file.split("L0", 1))
        # The output directory applies to a bare name; an -o with a directory
        # in it is used as given (it was cut to its name and written into the
        # input folder, #248)
        if output_directory and (args.out_file is None or Path(f_output).parent == Path(".")):
            f_output = output_directory + "/" + Path(f_output).name
        return f_output

    # The level-0 efield files are the input (tefield_l0 is read).  Earlier
    # outputs are removed before the folder is opened: a rerun, or a folder
    # already holding L1 files (as the committed sample does), failed with
    # NotUniqueEvent, and a file still open in the DataDirectory is reused
    # by name when reopened (#231).
    input_files = sorted(glob.glob(os.path.join(args.directory, "efield_*_L0_*.root")))
    if not input_files:
        raise SystemExit("GRANDlib: convert_efield2efield: %s has no efield file (efield_*_L0_*.root)"
                         % args.directory)
    # The trees must exist and agree before anything is computed or removed:
    # they failed deep inside, or were accepted silently (#249)
    from grand.dataio.consistency import check_event_trees
    checked = grand.dataio.DataDirectory(args.directory)
    try:
        if checked.tshower_l0 is None:
            raise SystemExit("GRANDlib: convert_efield2efield: %s has no shower file "
                             "(shower_*_L0_*.root)" % args.directory)
        check_event_trees(checked.tefield_l0, checked.trun_l0, "convert_efield2efield", args.directory)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
    finally:
        checked.close()

    outputs = [efield_output(f) for f in input_files]
    run_outputs = [level1_name(f, output_directory)
                   for pattern in ("run_*_L0_*.root", "runefieldsim_*_L0_*.root")
                   for f in glob.glob(os.path.join(args.directory, pattern))[:1]]
    for path in dict.fromkeys(outputs + run_outputs):
        if os.path.exists(path):
            logger.info(f"replacing the existing {path}")
            os.remove(path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    #Open file
    d_input = grand.dataio.DataDirectory(args.directory)

    # Loop through the level-0 efield files
    for f_input_file, f_output in zip(input_files, outputs):

        df_input_file = grand.dataio.DataFile(f_input_file)
        tefield = df_input_file.tefield_l0
        entries = tefield.get_number_of_entries()

        #Get the trees
        trun = d_input.trun_l0
        trunefieldsim = d_input.trunefieldsim_l0
        tshower = d_input.tshower_l0
        # tefield = d_input.tefield_l0


        #get the list of events
        events_list=tefield.get_list_of_events()

        nb_events = len(events_list)

        # If there are no events in the file, exit: this logged "Exiting." and
        # went on to write empty output files (#248)
        if nb_events == 0:
          message = "There are no events in the file! Exiting."
          logger.error(message)
          raise SystemExit("GRANDlib: convert_efield2efield: %s has no events" % f_input_file)

        # Opened only once there is something to write
        logger.info(f"save result in {f_output}")
        out_tefield = grand.dataio.TEfield(f_output)

        ####################################################################################
        # start looping over the events
        ####################################################################################
        previous_run=None
        for event_idx in range(nb_events):

           event_number = events_list[event_idx][0]
           run_number = events_list[event_idx][1]
           assert isinstance(event_number, int)
           assert isinstance(run_number, int)
           logger.info(f"Running on event_number: {event_number}, run_number: {run_number}")

           tefield.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
           tshower.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
           if previous_run != run_number:                        # load only for new run.
                trun.get_run(run_number)                         # update run info to get site latitude and longitude.
                trunefieldsim.get_run(run_number)
                previous_run = run_number
           # A lookup that finds nothing leaves the previous entry loaded:
           # the event would get another event's shower or run (issue #247).
           if (int(tshower.event_number), int(tshower.run_number)) != (event_number, run_number):
                raise KeyError(f"GRANDlib: convert_efield2efield: the shower tree has no entry for event "
                               f"{event_number} of run {run_number}; the efield and shower files do not match")
           if int(trun.run_number) != run_number:
                raise KeyError(f"GRANDlib: convert_efield2efield: the run tree has no entry for run "
                               f"{run_number}; the efield and run files do not match")

            # stack efield traces
           trace_shape = np.asarray(tefield.trace).shape  # (nb_du, 3, tbins of a trace)
           du_id = np.asarray(tefield.du_id)         # used for printing info and saving in voltage tree.
           event_dus_indices = tefield.get_dus_indices_in_run(trun)
           nb_du = trace_shape[0]
           sig_size = trace_shape[-1]
           traces = np.asarray(tefield.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)

           # A shower that hit no antenna is kept, empty, as at every other
           # level (#91); it crashed here on f_samp_mhz[0] (#248)
           if len(du_id) == 0:
              out_tefield.run_number = tefield.run_number
              out_tefield.event_number = tefield.event_number
              out_tefield.du_count = 0
              out_tefield.du_id = []
              out_tefield.trace = np.zeros((0, 3, 0), dtype=np.float32)
              out_tefield.du_seconds = []
              out_tefield.du_nanoseconds = []
              out_tefield.trigger_position = []
              out_tefield.analysis_level = tefield.analysis_level+1
              out_tefield.fill()
              out_tefield.write()
              logger.warning(f"Event {event_number} of run {run_number} has no antenna; written empty")
              continue


           dt_ns = np.asarray(trun.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns.
           f_samp_mhz = 1e3/dt_ns                                 # MHz

           #i recover the original values becouse this variables are rewriten (poor programing here on my side)
           target_duration_us = args.target_duration_us           # if different from 0, will adjust padding factor to get a trace of this lenght in us
           target_sampling_rate_mhz = args.target_sampling_rate_mhz

           if target_sampling_rate_mhz==f_samp_mhz[0]:
             target_sampling_rate_mhz=0 #no need to resample


           if(target_duration_us>0):                              #MATIAS: TODO: Here we lost the capability to use different sampling rates on different antennas!.
              target_lenght= int(target_duration_us*f_samp_mhz[0])
              padding_factor=target_lenght/sig_size
              logger.debug(f"padding factor adjusted to {padding_factor} to reach a duration of {target_duration_us} us at {f_samp_mhz[0]} Mhz -> {target_lenght} samples")
           else:
              target_lenght=int(padding_factor * sig_size + 0.5) #add 0.5 to avoid any rounding error for the int conversion
              target_duration_us = target_lenght/f_samp_mhz[0]

           if not padding_factor >= 1:      # a bare AssertionError before (#233)
              raise SystemExit("GRANDlib: convert_efield2efield: --target_duration_us %s is shorter than the "
                               "traces (%.4g us); the output cannot be shorter than the input"
                               % (args.target_duration_us, sig_size / f_samp_mhz[0]))


           # common frequencies for all processing in Fourier domain. Fourier transform algorithms work better if the lenght of the trace is a multiple of 2,3 or 5
           fast_fft_size, fast_freqs_mhz = get_fastest_size_fft(sig_size, f_samp_mhz, padding_factor)

           logger.debug(f"Electric field lenght is {sig_size} samples at {f_samp_mhz[0]}Mhz, spanning {sig_size/f_samp_mhz[0]} us.")
           logger.debug(f"Applying a padding factor of {padding_factor} we will take it to {target_lenght} samples, spanning {target_lenght/f_samp_mhz[0]} us.")
           logger.debug(f"The optimal number of frequency bins to do a fast fft is {len(fast_freqs_mhz)} giving traces of {fast_fft_size} samples.")
           logger.debug(f"With this we will obtain traces spanning {fast_fft_size/f_samp_mhz[0]} us, that we will then truncate if needed to get the requested trace duration.")


           # container to collect computed Voc and the final voltage in time domain for one event.
           vout = np.zeros((trace_shape[0], trace_shape[1], fast_fft_size), dtype=float) # time domain
           vout_f = np.zeros((trace_shape[0], trace_shape[1], len(fast_freqs_mhz)), dtype=np.complex64) # frequency domain

           #####################################################
           # Prepare the filter.  The idea was to filter in the frequency domain, buti cant get it working properly for now, so i do it in the time domain
           ######################################################
           if(filter):
               # Sampling frequency in Hz
               Fs = f_samp_mhz[0]*1e6
               # Pass band frequency in Hz
               fp = np.array([50e6, 200e6])
               # Stop band frequency in Hz
               fs = np.array([30e6, 250e6])
               # Pass band ripple in dB
               Ap = 0.4
               # Stop band attenuation in dB
               As = 50
               # Compute pass band and stop band edge frequencies w.r.t. Nyquist rate
               # Normalized passband edge frequencies
               wp = fp/(Fs/2)
               # Normalized stopband edge frequencies
               ws = fs/(Fs/2)
               # Compute order of the elliptic filter
               N, wc = ss.ellipord(wp, ws, Ap, As)
               # Design digital elliptic bandpass filter
               sos = ss.ellip(N, Ap, As, wc, 'bandpass',output="sos")
               #plot filter response
               if(PLOT):
                b, a = ss.ellip(N, Ap, As, wc, 'bandpass')
                mfreqz(b, a, Fs)
                impz(b, a)
               logger.info(f"We are going to apply a band-passs elliptic filter of order {N} between {fp[0]/1e6} and {fp[1]/1e6} Mhz.")


           # we initialize the random seed
           if seed >= 0:  # 0 seeds too; -1 is the 'no seed' placeholder (#230)
             np.random.seed(seed*(event_idx+1))

           for du_idx in range(nb_du):

                #MATIAS: TODO:Each of this steps should be made a capability of the Efield trace class . Im sure this can be done withouth the loop.
                #this could also be done all in the same line, but for clarity i will put it step by step

                #extend the trace to the fast fft size
                tracex=np.pad(traces[du_idx,0],(0,fast_fft_size-len(traces[du_idx,0])),'constant')
                tracey=np.pad(traces[du_idx,1],(0,fast_fft_size-len(traces[du_idx,1])),'constant')
                tracez=np.pad(traces[du_idx,2],(0,fast_fft_size-len(traces[du_idx,1])),'constant')

                #add the calibration noise
                if(calsigma>0):
                  calfactor=np.random.normal(1,calsigma)
                  tracex=tracex*calfactor
                  tracey=tracey*calfactor
                  tracez=tracez*calfactor
                  logger.debug(f"Antenna {du_idx} smearing calibration factor {calfactor}")

                if(event_idx==0 and du_idx==PlotAnt and PLOT):
                  plt.plot(traces[du_idx, 0],label="original",linewidth=5)
                  plt.plot(tracex,label="paded")

                #add noise
                if(noise>0):
                    tracex=tracex + np.random.normal(0,noise,size=np.shape(tracex))
                    tracey=tracey + np.random.normal(0,noise,size=np.shape(tracey))
                    tracez=tracez + np.random.normal(0,noise,size=np.shape(tracez))

                if(event_idx==0 and du_idx==PlotAnt and PLOT):
                  plt.plot(tracex,label="noised")

               # test squarefilter for tim, this code can be removed if you are reading this after 2025
               # if(event_idx==0 and du_idx==PlotAnt and PLOT):
               #   #now, we compute the fourier transforms to use them in the resampling
               #   e_trace = coord.CartesianRepresentation( x=tracex, y=tracey, z=tracez,)
               #
               #   efield_idx = ElectricField(np.arange(0,len(e_trace.x)) * 1e-9, e_trace)
               #   fft_e = efield_idx.get_fft(fast_fft_size)
               #
               #   vout_f[du_idx, 0]=fft_e[0]
               #   vout_f[du_idx, 1]=fft_e[1]
               #
               #   mask=(fast_freqs_mhz>50) & (fast_freqs_mhz<200)
               #   vout_square=vout_f
               #   vout_square[:,:,~mask]=0
               #
               #   vout_f[du_idx, 2]=fft_e[2]
               #
               #   vout_square = sf.irfft(vout_square)
               #
               #   plt.plot(vout_square[PlotAnt][0],label="squarefilter")

                #filter
                if(filter):
                    tracex=ss.sosfilt(sos,tracex)
                    tracey=ss.sosfilt(sos,tracey)
                    tracez=ss.sosfilt(sos,tracez)

                    if(event_idx==0 and du_idx==PlotAnt):
                      plt.plot(tracex,label="filtered",linewidth=3)


                #now, we compute the fourier transforms to use them in the resampling
                e_trace = coord.CartesianRepresentation( x=tracex, y=tracey, z=tracez,)

                efield_idx = ElectricField(np.arange(0,len(e_trace.x)) * 1e-9, e_trace)
                fft_e = efield_idx.get_fft(fast_fft_size)

                vout_f[du_idx, 0]=fft_e[0]
                vout_f[du_idx, 1]=fft_e[1]
                vout_f[du_idx, 2]=fft_e[2]


                """
                #This is another way of cutting and decimating the trace. However, np.decimate aplies a filter and the result is not exactly the same.
                #this is just to test if np.decimate does a good job. You can remove this code if you are reading this in 2025  
                #cut to the target lenght
                            
                if(target_lenght<len(tracex)):
                  tracex=tracex[0:target_lenght]
                if(target_lenght<len(tracey)):
                  tracey=tracey[0:target_lenght]
                if(target_lenght<len(tracez)):
                  tracez=tracez[0:target_lenght]
                
                #if(event_idx==0 and du_idx==PlotAnt):  
                  #plt.plot(tracex,label="shortened")
    
                #decimate
                ratio=(f_samp_mhz[0]/target_sampling_rate_mhz) 
                print(ratio)
                assert ratio%1==0
                
                tracex=ss.decimate(tracex,int(ratio))
                tracey=ss.decimate(tracey,int(ratio))
                tracez=ss.decimate(tracez,int(ratio))
                
                if(event_idx==0 and du_idx==PlotAnt):
                  plt.scatter(np.arange(0,len(tracex))*ratio,tracex,label="decimated")
                """

           #here we do the resampling. #MATIAS: TODO This should be a funtion part of ElectricField class. We should be able to call ElectricField.Resample(new_sampling_rate)
           if(target_sampling_rate_mhz>0): #if we need to resample
                #compute new number of points
                ratio=(target_sampling_rate_mhz/f_samp_mhz[0])
                m=int(fast_fft_size*ratio)
                #now, since we resampled,  we have a new target_lenght
                target_lenght= int(target_duration_us*target_sampling_rate_mhz)
                logger.info(f"resampling the efield from {f_samp_mhz[0]} to {target_sampling_rate_mhz} MHz, new trace lenght is {target_lenght} samples, ratio is {ratio}")
                #MATIAS: TODO: now, we are missing a place to store the new sampling rate!
           else:
              m=fast_fft_size
              ratio=1

           #to resample we use fourier interpolation, becouse it seems to be better than scipy.decimate (points are closer to the original trace)
           vout = sf.irfft(vout_f, m)*ratio #renormalize the amplitudes.

           if(event_idx==0 and PLOT):
             plt.scatter(np.arange(0,len(vout[PlotAnt][0]))/ratio,vout[PlotAnt][0],label="sampled",c="red")


           #here we do the truncation #MATIAS: TODO This should be a funtion part of ElectricField class. We should be able to call ElectricField.Resize(new_size)
           if(target_lenght<np.shape(vout)[2]):
                logger.debug(f"truncating output to {target_lenght} samples")
                vout=vout[..., :target_lenght]
                if(event_idx==0 and PLOT):
                   plt.scatter(np.arange(0,len(vout[PlotAnt][0]))/ratio,vout[PlotAnt][0],label="sampled and shortened",c="green")

           if(event_idx==0 and PLOT):
             plt.legend()
             plt.show()

           #here i shuffle the time
           du_nanoseconds=np.asarray(tefield.du_nanoseconds)
           du_seconds=np.asarray(tefield.du_seconds)

           if(jitter>0):
               logger.info(f"adding {jitter} ns of time jitter to the trigger times.")
               #reinitialize the random number
               if seed >= 0:  # 0 seeds too; -1 is the 'no seed' placeholder (#230)
                 np.random.seed(seed*(event_idx+1))
               delays=np.round(np.random.normal(0,jitter,size=np.shape(du_nanoseconds)).astype(int))
               du_nanoseconds=du_nanoseconds+delays

               #now we have to roll the seconds
               maskplus= du_nanoseconds >=1e9
               maskminus= du_nanoseconds <0
               du_nanoseconds[maskplus]-=int(1e9)
               du_seconds[maskplus]+=int(1)
               du_nanoseconds[maskminus]+=int(1e9)
               du_seconds[maskminus]-=int(1)

               if(PLOT):
                 plt.plot(delays)
                 plt.show()

           #Finally we save to a file
           #MATIAS: TODO: There shoudl be a way of coping all the field from tefield, and then modifing only what i want. There is, but it blows up in segfault
           #see issue #81 https://github.com/grand-mother/grand/issues/81
           #out_tefield.copy_contents(tefield)

           out_tefield.run_number = tefield.run_number
           out_tefield.event_number = tefield.event_number
           out_tefield.du_id = tefield.du_id
           # Was never set, so every event read as having no antenna (#248)
           out_tefield.du_count = len(du_id)

           out_tefield.trace=vout
           out_tefield.du_nanoseconds=du_nanoseconds
           out_tefield.du_seconds=du_seconds

           out_tefield.analysis_level = tefield.analysis_level+1

            #modify the trigger position if needed
           if(target_sampling_rate_mhz>0):
              originalsampling=f_samp_mhz[0]
              newsampling=target_sampling_rate_mhz
              ratio=originalsampling/newsampling
           else:
              ratio=1.0

           out_tefield.trigger_position=np.ushort(np.asarray(tefield.trigger_position)/ratio)

           out_tefield.fill()
           out_tefield.write()


    #TODO: Ask Lech how to do this for files with multiple runs.
    #now, we copy trun and change the sampling rate (filename to be changed when sim2root changes)
    #f_output = d_input.ftefield.filename.replace("L0", "L1")
    # Into -od, with the efield files: they went into the input folder, so
    # the -od folder could not feed the next step (#231)
    filename = level1_name(glob.glob(args.directory+ "/run_*L0*.root")[0], output_directory)
    outrun = grand.dataio.TRun(filename)
    outrun.copy_contents(trun)
    if(target_sampling_rate_mhz>0):
      outrun.t_bin_size = [1e3/target_sampling_rate_mhz]*len(outrun.t_bin_size) 
    outrun.analysis_level = trun.analysis_level+1
    outrun.fill()
    outrun.write()




    #TODO: if we changed the trace lenght this needs to update t_post acordingly.        
    #now, we copy trunefieldsim and change tpost (filename to be changed when sim2root changes)
    #f_output = d_input.ftefield.filename.replace("L0", "L1")
    filename = level1_name(glob.glob(args.directory+ "/runefieldsim_*L0*.root")[0], output_directory)
    outrunefieldsim = grand.dataio.TRunEfieldSim(filename)
    outrunefieldsim.copy_contents(trunefieldsim)
    outrunefieldsim.analysis_level = trunefieldsim.analysis_level+1
    
    outrunefieldsim.fill()
    outrunefieldsim.write()



    # =============================================
    logger.info(mlg.string_end_script())
