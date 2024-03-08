#! /usr/bin/env python3
"""
Script to to plot traces  provided in GRAND format.
   
usage: illustrateSimPipe.py directory [-h]


Calculation of Hardware-like Efield input file.

positional arguments:
  directory             Simulation output data directory in GRANDROOT format.

optional arguments:
  -h, --help            show this help message and exit
  --verbose {debug,info,warning,error,critical}
 

March 2024, M Tueros. On a Train to Toulouse.
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
        "--verbose",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logger verbosity.",
    )    
    # retrieve argument
    return parser.parse_args()



#plot sinals 1,2,3,4 and their fourier transforms, as well as signal4 sampled at sampling rate 
import scipy.fft as sfft
def plotfigure(time1="Off",signal1="Off",srate1=0,time2="Off",signal2="Off",srate2=0,time3="Off",signal3="Off",srate3=0,time4="Off",signal4="Off",srate4=0,label1="1",label2="2",label3="3",label4="3",Freqlimit=0.5):

    '''
    This will do the fft of the signals, and plot the signals and the fft.
    time should be in ns 
    srate is the sampling rate, in Ghz
    '''
    print(srate1,srate2,srate3,srate4)
    #fig 1 and fig 2 magic
    fig,ax=plt.subplots(2,1,figsize=(12,9))

    if(signal1 != "Off"):
      ax[0].plot(time1,signal1,label=label1,color="blue")
    if(signal2 != "Off"):
      ax[0].plot(time2,signal2,label=label2,color="orange")
    if(signal3 != "Off"):
      ax[0].plot(time3,signal3,label=label3,color="black")
    if(signal4 != "Off"):
      ax[0].plot(time4,signal4,label=label4,color="green",linewidth=3)
      
    # Set the separation of the labels in the x-axis every 10 units
    #ax[0].set_xticks(np.arange(np.min(time), np.max(time)+1, step=10/ts)) 
    # Make the vertical grid visible and semitransparent
    ax[0].grid(visible=True, axis="x", alpha=0.5)
    ax[0].axvline(800,linestyle="dashed",color="black")
    ax[0].set_ylabel('Amplitude (uV/m or uV) ',fontsize=14)
    ax[0].set_xlabel('Time (ns)',fontsize=14)
    #ax[0].set_xlim(0,max(time))    
   
    #if i want to put everything in the same plot it might be nice to have all the ffts with the same number of frequencies? 
     
    #plot in frequency        
    if(signal1 != "Off"):
      signal1_s=sfft.rfft(signal1)/(len(signal1))
      signal1_f=sfft.rfftfreq(len(signal1))*srate1  #signal spectrum frequencies is in units of nyquist rate, so this must be multipkied by srate/2
      ax[1].plot(signal1_f,np.abs(signal1_s),label=label1,color="blue")  
    if(signal2 != "Off"):
      signal2_s=sfft.rfft(signal2)/(len(signal2))
      signal2_f=sfft.rfftfreq(len(signal2))*srate2 #signal spectrum frequencies
      ax[1].plot(signal2_f,np.abs(signal2_s),label=label2,color="orange")  
    if(signal3 != "Off"):
      signal3_s=sfft.rfft(signal3)/(len(signal3))
      signal3_f=sfft.rfftfreq(len(signal3))*srate3          #signal spectrum frequencies
      ax[1].plot(signal3_f,np.abs(signal3_s),label=label3,color="black")  
    if(signal4 != "Off"):
      signal4_s=sfft.rfft(signal4)/(len(signal4))
      signal4_f=sfft.rfftfreq(len(signal4))*srate4  #signal spectrum frequencies
      ax[1].plot(signal4_f,np.abs(signal4_s),label=label4,color="green", linewidth=3)
       
    ax[1].axvline(srate1/2,linestyle="dashed",color="blue")
    ax[1].axvline(srate2/2,linestyle="dashdot",color="orange")
    ax[1].axvline(srate3/2,linestyle="dashed",color="black")
    ax[1].axvline(srate4/2,linestyle="dashdot",color="green")
    ax[1].axvline(0.05,linestyle="dotted",color="red")
    ax[1].axvline(0.2,linestyle="dotted",color="red")

    ax[1].set_ylabel('|Amplitude| (AU)',fontsize=14)
    ax[1].set_xlabel('Frequency (GHz)',fontsize=14)
    ax[1].grid(visible=True, axis="y", alpha=0.5)
    ax[1].set_xlim(-0.02,0.27)
    ax[1].legend()    
    fig.set_tight_layout(True)
    return fig,ax




if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt  
    import glob  
    import grand.manage_log as mlg
    import grand.dataio.root_trees as groot  
    plt.rcParams.update({'font.size': 16})

    
    ################# Manage logger and input arguments ######################################   
    logger = mlg.get_logger_for_script(__name__)
    args = manage_args()
    print(args.verbose) 
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    logger.info("Displaying events on directory "+args.directory)    
    
    #############################################################################################
    #############################################################################################
    #Open file
    d_input = groot.DataDirectory(args.directory)
    
    #Get the trees L0
    trun_l0 = d_input.trun_l0
    tshower_l0 = d_input.tshower_l0
    tefield_l0 = d_input.tefield_l0
    tvoltage_l0 = d_input.tvoltage_l0           
    #Get the trees L1
    tefield_l1 = d_input.tefield_l1
    tadc_l1 = d_input.tadc_l1
    trun_l1 = d_input.trun_l1
   
   
    #now open them manually until lech solves the issue
    '''
    print(d_input.ftefield_l0.filename)
    tefield_l0=groot.TEfield(d_input.ftefield_l0.filename)    
    print(d_input.ftefield_l1.filename)
    tefield_l1=groot.TEfield(d_input.ftefield_l1.filename)

    print(d_input.ftvoltage_l0.filename)    
    tvoltage_l0=groot.TVoltage(d_input.ftvoltage_l0.filename)

    print(d_input.ftadc_l1.filename)
    tadc_l1=groot.TADC(d_input.ftadc_l1.filename)

    print(d_input.ftrun_l0.filename)
    trun_l0=groot.TRun(d_input.ftrun_l0.filename)
    print(d_input.ftrun_l1.filename)
    trun_l1=groot.TRun(d_input.ftrun_l1.filename)
    '''


    #get the list of events
    events_list=tefield_l1.get_list_of_events()

    nb_events = len(events_list)

    # If there are no events in the file, exit
    if nb_events == 0:
      message = "There are no events in the file! Exiting."
      logger.error(message)
      sys.exit()
      
    ####################################################################################
    # start looping over the events
    ####################################################################################
    previous_run=None    
    for event_number,run_number in events_list:
    
       #event_number = events_list[event_idx][0]
       #run_number = events_list[event_idx][1]
       assert isinstance(event_number, int)
       assert isinstance(run_number, int)
       logger.info(f"Running on event_number: {event_number}, run_number: {run_number}")
    
       
       tefield_l0.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
       tshower_l0.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
       tefield_l1.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
       tvoltage_l0.get_event(event_number, run_number)
       tadc_l1.get_event(event_number, run_number)
       
       #TODO: WRITE AN ISSUE. Ask Lech Would it be posible that get_run automatically does nothing if you ask for the run its currently being pointed to?
       if previous_run != run_number:                        # load only for new run.
            trun_l0.get_run(run_number)                         # update run info to get site latitude and longitude.
            trun_l1.get_run(run_number)                         # update run info to get site latitude and longitude.            
            previous_run = run_number
       
       #
       trace = np.asarray(tefield_l0.trace, dtype=np.float32)   # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
       vtrace = np.asarray(tvoltage_l0.trace, dtype=np.float32)
       atrace= np.asarray(tadc_l1.trace_ch, dtype=np.float32)
       rtrace= np.asarray(tefield_l1.trace, dtype=np.float32)
       du_id = np.asarray(tefield_l0.du_id)                     # used for printing info and saving in voltage tree.
       
       #TODO: this forces an homogeneous antenna array.
       trace_shape = trace.shape  # (nb_du, 3, tbins of a trace)
       nb_du = trace_shape[0]
       sig_size = trace_shape[-1]
       logger.info(f"Event has: {nb_du} detector unists, with a signal size of: {sig_size}")
       
       #this gives the indicies of the antennas of the array participating in this event, 
       event_dus_indices = tefield_l0.get_dus_indices_in_run(trun_l0)       
       print(event_dus_indices )

       dt_ns_l0 = np.asarray(trun_l0.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 
       dt_ns_l1 = np.asarray(trun_l1.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 
              
       #print(dt_ns_l0,dt_ns_l1)
       du_xyzs= np.asarray(trun_l0.du_xyz)[event_dus_indices] # antenna positions in shc? coordinates (TODO: Shouldnt this be in grand coordinates?        
       #print(du_xyzs)
       
       #anyway, this will loop over all stations.          
       for du_idx in range(nb_du):
         print(du_idx)
         #unstack the trace
        
         #I only plot the first quarter of the trace becouse that is where the peak is, and adding the long tail just puts a lot of noise on the fft,
         
         tracex=trace[du_idx,0]*10
         tracey=trace[du_idx,1]*10
         tracez=trace[du_idx,2]*10
         tracet=np.arange(0,len(tracez))*dt_ns_l0[du_idx] #+1200*dt_ns_l0[du_idx]
         #print("efield",len(tracez))

         vtracex=vtrace[du_idx,0]
         vtracey=vtrace[du_idx,1]
         vtracez=vtrace[du_idx,2]
         vtracet=np.arange(0,len(vtracez))*dt_ns_l0[du_idx] #+1200*dt_ns_l0[du_idx]
         #print("voltage",len(vtracez))

         atracex=atrace[du_idx,0]*109.86      #this number comes from doing 0.9V/8192, the maximum of ADC divided by 13bits
         atracey=atrace[du_idx,1]*109.86
         atracez=atrace[du_idx,2]*109.86
         atracet=np.arange(0,len(atracez))*dt_ns_l1[du_idx] #+300*dt_ns_l1[du_idx]
         #print("adc",len(atracez))

         rtracex=rtrace[du_idx,0]*10
         rtracey=rtrace[du_idx,1]*10
         rtracez=rtrace[du_idx,2]*10
         rtracet=np.arange(0,len(rtracez))*dt_ns_l1[du_idx] #+300*dt_ns_l1[du_idx]          
         #print("refield",len(rtracez))
                
         fig,ax=plotfigure(time1=tracet,signal1=tracex,srate1=1/dt_ns_l0[du_idx],time2=vtracet,signal2=vtracex,srate2=1/dt_ns_l0[du_idx],time3=atracet,signal3=atracex,srate3=1/dt_ns_l1[du_idx],time4=rtracet,signal4=rtracex,srate4=1/dt_ns_l1[du_idx],label1="efield_l0 x10",label2="voltage_l0",label3="adc_l1 x110",label4="efield_l1 x10",Freqlimit=0.5)
         plt.show()     
