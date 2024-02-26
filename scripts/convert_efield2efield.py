#! /usr/bin/env python3
"""
Script to compute a "hardwarelike" efield from electric field.
Electric field traces are provided in a ROOT file.

To Run:
    python convert_efield2voltage.py <efield.root> -o <output.root> # RF chain and noise added automatically.
    python convert_efield2voltage.py <efield.root> -o <output.root> --seed 0 
    convert_efield2voltage.py <efield.root> -o <output.root> --no_noise --filter

In this file:

    Options that can be given to compute_voltage() depending on what you want to compute.
        Compute/simulate voltage for any or all DUs for any or all events in input file.

        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)). If None, all events in an input file is used.
        :    type: int, list, np.ndarray
        :param du_idx: index of DU for which voltage is computed. If None, all DUs of an event is used. du_idx can be used for only one event.
        :    type: int, list, np.ndarray
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.  If None, all events in an input file is used.
        :    type: int, list, np.ndarray
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.  If None, all events in an input file is used.
        :    type: int, list, np.ndarray  

        Note: Either event_idx, or both event_number and run_number must be provided, or all three must be None.      
              if du_idx is provided, voltage of the given DU of the given event is computed. 
              du_idx can be an integer or list/np.ndarray. du_idx can be used for only one event.
              If improper event_idx or (event_number and run_number) is used, an error is generated when self.get_event() is called.
              Selective events with either event_idx or both event_number and run_number can be given.
              If list/np.ndarray is provided, length of event_number and run_number must be equal.    



Feb 2024, M Tueros
"""
def check_float_day_hour(s_hour):
    f_hour = float(s_hour)
    if f_hour < 0 or f_hour > 24:
        raise argparse.ArgumentTypeError(f"lts must be > 0h and < 24h.")
    return f_hour


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
        "--no_noise",
        action="store_false",
        default=True,
        help="don't add galactic noise.",
    )
    parser.add_argument(
        "--filter",
        action="store_false",
        default=True,
        help="don't add the filter.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default=None,
        help="output file in GRANDROOT format. If the file exists it is overwritten.",
    )
    parser.add_argument(
        "-od",
        "--out_directory",
        default=None,
        help="output directory in GRANDROOT format. If not given, is it the same as input directory",
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
        "--lst",
        type=check_float_day_hour,
        default=18.0,
        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h for the EW arm.",
    )
    parser.add_argument(
        "--padding_factor",
        type=float,
        default=1.0,
        help="Increase size of signal with zero padding, with 1.2 the size is increased of 20%%. ",
    )
    parser.add_argument(
        "--target_duration_us",
        type=float,
        default=0,
        help="Adujust (and override) padding factor in order to get a signal of the given duration, in us",
    )    
    parser.add_argument(
        "--target_sampling_rate_mhz",
        type=float,
        default=0,
        help="Target sampling rate of the data in Mhz",
    )      
    # retrieve argument
    return parser.parse_args()


def get_time_samples(tefield,nb_du,sig_size): #note that gets the time from t0, you have to substract t_pre to get the actual time
    """
    Define time sample in ns for the duration of the trace
    t_samples.shape  = (nb_du, sig_size)
    t_start_ns.shape = (nb_du,)
    """
    t_start_ns = np.asarray(tefield.du_nanoseconds)[...,np.newaxis]   # shape = (nb_du, 1)
    t_samples = (
        np.outer(
            dt_ns * np.ones(nb_du), np.arange(0, sig_size, dtype=np.float64)
            ) + t_start_ns )     
    logger.debug(f"shape du_nanoseconds and t_samples =  {t_start_ns.shape}, {t_samples.shape}")

    return t_samples


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
    freqs_mhz = sf.rfftfreq(fast_size, dt_s[0]) * 1e-6
    #print(f"padding_factor {padding_factor} sig_size {sig_size} ({padding_factor * sig_size +0.5}) fast size {fast_size} freqs_mhz size {len(freqs_mhz)}")
    return fast_size, freqs_mhz


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
    from typing import Union
    import numpy as np
    from pathlib import Path
    import grand.manage_log as mlg
    import grand.dataio.root_trees as groot
    import grand.geo.coordinates as coord
    from grand.basis.type_trace import ElectricField        
    import scipy.fft as sf
    import scipy.signal as ss
    import matplotlib.pyplot as plt    

    ################# Manage logger and input arguments ######################################
    
    logger = mlg.get_logger_for_script(__name__)
    args = manage_args()
    print(args.verbose) 
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    logger.info("Computing electric field from the input electric field.")

    # If no output directory given, define it as input directory
    if args.out_directory is None:
        args.out_directory = args.directory

    seed = None if args.seed==-1 else args.seed
    logger.info(f"seed used for random number generator is {seed}.")

    no_noise = args.no_noise
    filter = args.filter
    f_output=args.out_file
    output_directory=args.out_directory
    padding_factor=1

    target_sampling_rate_mhz = args.target_sampling_rate_mhz     
    assert  target_sampling_rate_mhz >= 0

    target_duration_us = args.target_duration_us       # if different from 0, will adjust padding factor to get a trace of this lenght in us        
    assert target_duration_us >= 0

    #############################################################################################


    #Open file
    d_input = groot.DataDirectory(args.directory)
    
    # If output filename given, use it
    if f_output:
       f_output = f_output
    # Otherwise, generate it from tefield filename
    else:
       f_output = d_input.ftefield.filename.replace("efield", "DC2efield")

    # If output directory given, use it
    if output_directory:
       f_output = output_directory + "/" + Path(f_output).name

    logger.info(f"save result in {f_output}")    
    out_tefield = groot.TEfield(f_output)

    
    #Get the trees
    trun = d_input.trun
    tshower = d_input.tshower
    tefield = d_input.tefield

    #get the list of events
    events_list=tefield.get_list_of_events()

    nb_events = len(events_list)

    # If there are no events in the file, exit
    if nb_events == 0:
      message = "There are no events in the file! Exiting."
      logger.error(message)

    previous_run=None
    
    for event_idx in range(nb_events):
       
       event_number = events_list[event_idx][0]
       run_number = events_list[event_idx][1]
       assert isinstance(event_number, int)
       assert isinstance(run_number, int)
       logger.info(f"Running on event_number: {event_number}, run_number: {run_number}")
       
       tefield.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
       tshower.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
       if previous_run != run_number:                      # load only for new run.
            trun.get_run(run_number)                         # update run info to get site latitude and longitude.
            previous_run = run_number
       
       
        # stack efield traces
       trace_shape = np.asarray(tefield.trace).shape  # (nb_du, 3, tbins of a trace)
       du_id = np.asarray(tefield.du_id)         # used for printing info and saving in voltage tree.
       event_dus_indices = tefield.get_dus_indices_in_run(trun)
       nb_du = trace_shape[0]
       sig_size = trace_shape[-1]
       traces = np.asarray(tefield.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)

       
       
       
       dt_ns = np.asarray(trun.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. #MATIAS: Why cast this to an array if it is a constant?
       f_samp_mhz = 1e3/dt_ns                                 # MHz                                             #MATIAS: this gets casted too!
       # comupte time samples in ns for all antennas in event with index event_idx.
       time_samples = get_time_samples(tefield,nb_du,sig_size)  # t_samples.shape = (nb_du, sig_size)

       target_sampling_rate_mhz = args.target_sampling_rate_mhz     
       target_duration_us = args.target_duration_us           # if different from 0, will adjust padding factor to get a trace of this lenght in us        
       
       if(target_duration_us>0):     
          target_lenght= int(target_duration_us*f_samp_mhz[0]) 
          padding_factor=target_lenght/sig_size 
          logger.debug(f"padding factor adjusted to {padding_factor} to reach a duration of {target_duration_us} us, {f_samp_mhz[0]} , {target_lenght}")                   
       else:
          target_lenght=int(padding_factor * sig_size + 0.5) #add 0.5 to avoid any rounding error for the int conversion
          target_duration_us = target_lenght/f_samp_mhz[0]
       
       print(padding_factor)   
       assert padding_factor >= 1  


       # common frequencies for all processing in Fourier domain.
       fft_size, freqs_mhz = get_fastest_size_fft(sig_size, f_samp_mhz, padding_factor)

       
       logger.info(f"Electric field lenght is {sig_size} samples at {f_samp_mhz[0]}, spanning {sig_size/f_samp_mhz[0]} us.")
       logger.info(f"With a padding factor of {padding_factor} we will take it to {target_lenght} samples, spanning {target_lenght/f_samp_mhz[0]} us.")
       logger.info(f"However, optimal number of frequency bins to do a fast fft is {len(freqs_mhz)} giving traces of {fft_size} samples.")
       logger.info(f"With this we will obtain traces spanning {fft_size/f_samp_mhz[0]} us, that we will then truncate if needed to get the requested trace duration.")       
       
       # container to collect computed Voc and the final voltage in time domain for one event.
       vout = np.zeros((trace_shape[0], trace_shape[1], fft_size), dtype=float) # time domain      
       vout_f = np.zeros((trace_shape[0], trace_shape[1], len(freqs_mhz)), dtype=np.complex64) # frequency domain


       #here it would go the filtering. The idea was to filter in the frequency domain, buti cant get it working properly for now, so i do it in the time domain
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
       # using signal.ellipord
       N, wc = ss.ellipord(wp, ws, Ap, As)
         
       # Print the order of the filter and cutoff frequencies
       print('Order of the filter=', N)
       print('Cut-off frequency=', wc,fs,Fs)
         
       # Design digital elliptic bandpass filter 
       # using signal.ellip() function
       sos = ss.ellip(N, Ap, As, wc, 'bandpass',output="sos")
       #plot filter response
       #b, a = ss.ellip(N, Ap, As, wc, 'bandpass')
       #mfreqz(b, a, Fs)
       #impz(b, a)
       
       np.random.seed(seed)
       for du_idx in range(nb_du):
       
            #this is  to use the fft capabailites of the Efield object in grandlib, but actually i could just do the fft myself here. Its more code.
            e_trace = coord.CartesianRepresentation(
            x=ss.sosfilt(sos,traces[du_idx, 0]+np.random.normal(0,22,size=2*np.shape(traces[du_idx,0]))),
            y=ss.sosfilt(sos,traces[du_idx, 1]+np.random.normal(0,22,size=2*np.shape(traces[du_idx,1]))),
            z=ss.sosfilt(sos,traces[du_idx, 2]+np.random.normal(0,22,size=2*np.shape(traces[du_idx,2]))),
            )
            
            if(event_idx==0 and du_idx==6):
                plt.plot(traces[du_idx, 0],label="original")
                plt.plot(e_trace[0],label="noise+filtered")
                #plt.plot(time_samples[du_idx],traces[du_idx, 0]+np.random.normal(0,22,size=np.shape(traces[du_idx,0])),label="original + noise")
                
            
        
            efield_idx = ElectricField(time_samples[du_idx] * 1e-9, e_trace)
            fft_e = efield_idx.get_fft(fft_size)
            
            vout_f[du_idx, 0]=fft_e[0]
            vout_f[du_idx, 1]=fft_e[1]
            vout_f[du_idx, 2]=fft_e[2]

   

       #here we do the resampling. #MATIAS: TODO This should be a funtion part of ElectricField class. We should be able to call ElectricField.Resample(new_sampling_rate) 
       if(target_sampling_rate_mhz>0): #if we need to resample
            #compute new number of points
            ratio=(target_sampling_rate_mhz/f_samp_mhz[0])        
            m=int(fft_size*ratio)
            #now, since we resampled,  we have a new target_lenght
            target_lenght= int(target_duration_us*target_sampling_rate_mhz)                                
            logger.info(f"resampling the efield from {f_samp_mhz[0]} to {target_sampling_rate_mhz} MHz, new trace lenght is {target_lenght} samples")                         
            #we use fourier interpolation, becouse its easy!
            vout = sf.irfft(vout_f, m)*ratio #renormalize the amplitudes
            #MATIAS: TODO: now, we are missing a place to store the new sampling rate!
            if(event_idx==0):
              plt.scatter(np.arange(0,target_lenght)/ratio,vout[6][0],label="sampled")
            
            
       #here we do the truncation #MATIAS: TODO This should be a funtion part of ElectricField class. We should be able to call ElectricField.Resize(new_size)          
       if(target_lenght<np.shape(vout)[2]):           
            logger.info(f"truncating output to {target_lenght} samples") 
            vout=vout[..., :target_lenght] 
            if(event_idx==0):
              plt.scatter(np.range(0,target_lenght)/ratio,vout[6][0],label="sampled and shortened")            
      
       plt.show()
       

       #here i shuffle the time
       du_nanoseconds=np.asarray(tefield.du_nanoseconds)
       du_seconds=np.asarray(tefield.du_seconds)
       np.random.seed(seed)
       delays=np.round(np.random.normal(0,5000,size=np.shape(du_nanoseconds)).astype(int))
       du_nanoseconds=du_nanoseconds+delays
       #print("delays",delays)
       #print("delay test",du_nanoseconds-np.asarray(tefield.du_nanoseconds))
       #print("before nano",du_nanoseconds)
       #print("before sec",du_seconds)
       #now we have to roll the seconds
       maskplus= du_nanoseconds >=1e9
       maskminus= du_nanoseconds <0
       du_nanoseconds[maskplus]-=int(1e9)
       du_seconds[maskplus]+=int(1)   
       du_nanoseconds[maskminus]+=int(1e9)
       du_seconds[maskminus]-=int(1)     
       #print("after",du_nanoseconds)
       #print("after",du_seconds)

       #here we save to a file
       #MATIAS: TODO: There shoudl be a way of coping all the field from tefield, and then modifing only what i want 

       out_tefield.run_number = tefield.run_number
       out_tefield.event_number = tefield.event_number
       out_tefield.du_id = tefield.du_id
       
       out_tefield.trace=vout
              
       out_tefield.du_nanoseconds=du_nanoseconds
       out_tefield.du_seconds=du_seconds

      
       out_tefield.fill()
       out_tefield.write()


    # =============================================
    logger.info(mlg.string_end_script())
