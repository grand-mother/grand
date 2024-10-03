#! /usr/bin/env python3
"""
Script to to plot traces provided in GRAND format.
   
usage: illustrateSimPipe.py directory [-h]

Calculation of hardware-like Efield input file.

positional arguments:
  directory             Simulation output data directory in GRANDROOT format.

optional arguments:
  -h, --help            show this help message and exit
  --verbose {debug,info,warning,error,critical}
 
authors: @mtueros @jelenakhlr
March 2024
"""

import grand.dataio.root_trees as groot 
# import the rest of the guardians of the galaxy:
import grand.manage_log as mlg
<<<<<<< HEAD
#import raw_root_trees as RawTrees # this is here in Common
=======
import raw_root_trees as RawTrees # this is here in Common
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
import sys
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re

# better plots
from matplotlib import rc
# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
rc('font', size = 11.0)

# grand logger
logger = mlg.get_logger_for_script(__name__)

def manage_args():
    parser = argparse.ArgumentParser(
        description="Calculation of Hardware-like Efield input file."
    )
    parser.add_argument(
        "directory",
        help="Simulation output data directory in GRANDROOT format."
    )
    parser.add_argument(
<<<<<<< HEAD
=======
        "--savefig_dir",
        type=str,
        default=None,
        help="Directory to save figures. Defaults to simulation output directory if not specified."
    )
    parser.add_argument(
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
        "--verbose",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logger verbosity."
    )
<<<<<<< HEAD
    parser.add_argument(
     "--savefig",
     action="store_true",
     default=False,
     help="save figures to files insted of displaying them."
     )
    parser.add_argument(
     "--sim",
     default="None",
     help="specify simulator: Coreas vs. Zhaires"
     )
    # retrieve argument
    return parser.parse_args()



=======
    parser.add_argument( 
        "--savefig",
        action="store_true",
        default=False,
        help="save figures to files insted of displaying them."
    )
    parser.add_argument(
        "--sim",
        default="None",
        help="specify simulator: Coreas vs. Zhaires"
        )
    # retrieve argument
    return parser.parse_args()

>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
def plot_core_positions(directory, t_0_shift=False):
  d_input = groot.DataDirectory(directory)

  #Get the trees L0
  tshower_l0 = d_input.tshower_l0

  #get the list of events
  events_list = tshower_l0.get_list_of_events()
  nb_events = len(events_list)

  # If there are no events in the file, exit
  if nb_events == 0:
    sys.exit("There are no events in the file! Exiting.")
  
  corex=np.zeros(nb_events)
  corey=np.zeros(nb_events)
<<<<<<< HEAD
  corez=np.zeros(nb_events)    
=======
  corez=np.zeros(nb_events)   

   
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
  ####################################################################################
  # start looping over the events
  ####################################################################################
  previous_run = None    
  i=0
<<<<<<< HEAD
  for event_number,run_number in events_list:
      assert isinstance(event_number, int)
      assert isinstance(run_number, int)
      logger.info(f"Running event_number: {event_number}, run_number: {run_number}")
=======
  myrun=0
  for event_number,run_number in events_list:
      assert isinstance(event_number, int)
      assert isinstance(run_number, int)
      logger.debug(f"Running event_number: {event_number}, run_number: {run_number}")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
      
      tshower_l0.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
      
      corex[i]=tshower_l0.shower_core_pos[0]
      corey[i]=tshower_l0.shower_core_pos[1]
      corez[i]=tshower_l0.shower_core_pos[2]
<<<<<<< HEAD
            
      i=i+1   

  plt.scatter(corex,corey)
  plt.show()
=======
      myrun=run_number      
      i=i+1   

  # Plot arrival time distribution
  # Create a figure with subplots to match the other plot 
  fig, ax1 = plt.subplots(1,1, figsize=(8, 6))
  
  map = ax1.scatter(corex, corey, marker='o', s=20)
  ax1.set_title(f"core positions")
  ax1.set_xlabel("X (northing) (m)")
  ax1.set_ylabel("Y (westing)(m)")  
  plt.tight_layout()
  if(args.savefig):
    plt.savefig(f"{plot_dir}/plots/CorePositions_{myrun}.png")
    plt.close(fig)
  else: 
    plt.show()


>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f

def plot_traces_all_levels(directory, t_0_shift=False):
  d_input = groot.DataDirectory(directory)

  #Get the trees L0
  trun_l0 = d_input.trun_l0
  trunefieldsim_l0=d_input.trunefieldsim_l0  
  tshower_l0 = d_input.tshower_l0
  tefield_l0 = d_input.tefield_l0
  tvoltage_l0 = d_input.tvoltage_l0           
  #Get the trees L1
  tefield_l1 = d_input.tefield_l1
  tadc_l1 = d_input.tadc_l1
  trun_l1 = d_input.trun_l1
  trunefieldsim_l1=d_input.trunefieldsim_l1

  #get the list of events
  events_list = tefield_l1.get_list_of_events()
  nb_events = len(events_list)

  # If there are no events in the file, exit
  if nb_events == 0:
    sys.exit("There are no events in the file! Exiting.")
  
  ####################################################################################
  # start looping over the events
  ####################################################################################
  previous_run = None    
  for event_number,run_number in events_list:
      assert isinstance(event_number, int)
      assert isinstance(run_number, int)
<<<<<<< HEAD
      logger.info(f"Running event_number: {event_number}, run_number: {run_number}")
=======
      logger.debug(f"Running event_number: {event_number}, run_number: {run_number}")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
      
      tefield_l0.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
      tshower_l0.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
      tefield_l1.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
      tvoltage_l0.get_event(event_number, run_number)
      tadc_l1.get_event(event_number, run_number)
      
      #TODO: WRITE AN ISSUE. Ask Lech Would it be posible that get_run automatically does nothing if you ask for the run its currently being pointed to?
      if previous_run != run_number:                          # load only for new run.
        trun_l0.get_run(run_number)                         # update run info to get site latitude and longitude.
        trun_l1.get_run(run_number)                         # update run info to get site latitude and longitude.            
        trunefieldsim_l0.get_run(run_number)
        trunefieldsim_l1.get_run(run_number)        
        previous_run = run_number
      
      
      trace_efield_L0 = np.asarray(tefield_l0.trace, dtype=np.float32)   # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
      trace_voltage = np.asarray(tvoltage_l0.trace, dtype=np.float32)
      trace_ADC_L1= np.asarray(tadc_l1.trace_ch, dtype=np.float32)
      trace_efield_L1= np.asarray(tefield_l1.trace, dtype=np.float32)
      du_id = np.asarray(tefield_l0.du_id) # MT: used for printing info and saving in voltage tree.
      # JK: du_id is currently unused - TODO


      # t0 calculations
      event_second = tshower_l0.core_time_s
      event_nano = tshower_l0.core_time_ns
      t0_efield_L1 = (tefield_l1.du_seconds-event_second)*1e9  - event_nano + tefield_l1.du_nanoseconds 
      t0_efield_L0 = (tefield_l0.du_seconds-event_second)*1e9  - event_nano + tefield_l0.du_nanoseconds
      t0_adc_L1 = (tadc_l1.du_seconds-event_second)*1e9  - event_nano + tadc_l1.du_nanoseconds
      t0_voltage_L0 = (tvoltage_l0.du_seconds-event_second)*1e9  - event_nano + tvoltage_l0.du_nanoseconds 

      #time window parameters. time windows go from t0-t_pre to t0+t_post
      t_pre_L0=trunefieldsim_l0.t_pre
      t_pre_L1=trunefieldsim_l1.t_pre


      #TODO: this forces a homogeneous antenna array.
      trace_shape = trace_efield_L0.shape  # (nb_du, 3, tbins of a trace)
      nb_du = trace_shape[0]
      sig_size = trace_shape[-1]
      logger.info(f"Event has {nb_du} DUs, with a signal size of: {sig_size}")
      
      #this gives the indices of the antennas of the array participating in this event
      event_dus_indices = tefield_l0.get_dus_indices_in_run(trun_l0)

      dt_ns_l0 = np.asarray(trun_l0.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 
      dt_ns_l1 = np.asarray(trun_l1.t_bin_size)[event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns. 

      #how to get tpre (the starting time of the trace) if you dont have trunefieldsim (on data). This also works with voltage and adc trees
      t_pre_L0_foralldu=tefield_l0.trigger_position*dt_ns_l0
      t_pre_L1_foralldu=tefield_l1.trigger_position*dt_ns_l1      

      du_xyzs= np.asarray(trun_l0.du_xyz)[event_dus_indices] 
     

      # loop over all stations.          
      for du_idx in range(nb_du):
<<<<<<< HEAD
        print(f"Running DU number {du_idx}")
=======
        logger.debug(f"Running DU number {du_idx}")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f

        # efield trace L0
        trace_efield_L0_x = trace_efield_L0[du_idx,0]
        trace_efield_L0_y = trace_efield_L0[du_idx,1]
        trace_efield_L0_z = trace_efield_L0[du_idx,2]
        trace_efield_L0_time = np.arange(0,len(trace_efield_L0_z)) * dt_ns_l0[du_idx] - t_pre_L0
        
        # voltage trace
        trace_voltage_x = trace_voltage[du_idx,0]
        trace_voltage_y = trace_voltage[du_idx,1]
        trace_voltage_z = trace_voltage[du_idx,2]
        trace_voltage_time = np.arange(0,len(trace_voltage_z)) * dt_ns_l0[du_idx] - t_pre_L0

        # adc trace
        trace_ADC_L1_x = trace_ADC_L1[du_idx,0]
        trace_ADC_L1_y = trace_ADC_L1[du_idx,1]
        trace_ADC_L1_z = trace_ADC_L1[du_idx,2]
        trace_ADC_L1_time = np.arange(0,len(trace_ADC_L1_z)) * dt_ns_l1[du_idx] - t_pre_L1
        
        # efield trace L1
        trace_efield_L1_x = trace_efield_L1[du_idx,0]
        trace_efield_L1_y = trace_efield_L1[du_idx,1]
        trace_efield_L1_z = trace_efield_L1[du_idx,2]
        trace_efield_L1_time = np.arange(0,len(trace_efield_L1_z)) * dt_ns_l1[du_idx] - t_pre_L1

        # time for plotting!
        # Create a figure with subplots
        if(args.savefig):
          fig, axs = plt.subplots(2,2, figsize=(8, 6))
        else:
          fig, axs = plt.subplots(2,2, figsize=(8, 6), sharex=True)
        if t_0_shift == True:
<<<<<<< HEAD
          print("shifting by t0")
=======
          logger.debug(f"shifting by t0")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
          trace_efield_L0_time += t0_efield_L0[du_idx]
          trace_voltage_time += t0_voltage_L0[du_idx]
          trace_ADC_L1_time += t0_adc_L1[du_idx]
          trace_efield_L1_time += t0_efield_L1[du_idx]

          plt.suptitle(f"event {event_number}, run {run_number}, antenna {du_idx} - Shower Time")
          savelabel = "with_t0_shift"
        else:
<<<<<<< HEAD
          print(f"NOT shifting by t0 - peaks should be at 0ns")
=======
          logger.debug(f"NOT shifting by t0 - peaks should be at 0ns")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
          plt.suptitle(f"event {event_number}, run {run_number}, antenna {du_idx} - Trigger Time")
          savelabel = "no_t0s_shift"
          

        # Plot voltage traces on the first subplot
        ax1=axs[0,0]
        ax1.plot(trace_voltage_time, trace_voltage_x, alpha=0.5, label="polarization N")
        ax1.plot(trace_voltage_time, trace_voltage_y, alpha=0.5, label="polarization E")
        ax1.plot(trace_voltage_time, trace_voltage_z, alpha=0.5, label="polarization v")
        ax1.set_title(f"voltage antenna {du_idx}")
        ax1.set_xlabel("time in ns")
        ax1.set_ylabel("voltage in uV")

        # adc traces
        ax2=axs[0,1]
        ax2.plot(trace_ADC_L1_time, trace_ADC_L1_x, alpha=0.5, label="polarization N")
        ax2.plot(trace_ADC_L1_time, trace_ADC_L1_y, alpha=0.5, label="polarization E")
        ax2.plot(trace_ADC_L1_time, trace_ADC_L1_z, alpha=0.5, label="polarization v")
        ax2.set_title(f"adc antenna {du_idx}")
        ax2.set_xlabel("time in ns")
        ax2.set_ylabel("counts")

        # Plot electric field L0
        ax3=axs[1,0]
        ax3.plot(trace_efield_L0_time, trace_efield_L0_x, alpha=0.5, label="polarization N")
        ax3.plot(trace_efield_L0_time, trace_efield_L0_y, alpha=0.5, label="polarization E")
        ax3.plot(trace_efield_L0_time, trace_efield_L0_z, alpha=0.5, label="polarization v")
        ax3.set_title(f"efield L0, antenna {du_idx}")
        ax3.set_xlabel("time in ns")
        ax3.set_ylabel("efield in uV/m")

        # Plot electric field L1
        ax4=axs[1,1]
        ax4.plot(trace_efield_L1_time, trace_efield_L1_x, alpha=0.5, label="polarization N")
        ax4.plot(trace_efield_L1_time, trace_efield_L1_y, alpha=0.5, label="polarization E")
        ax4.plot(trace_efield_L1_time, trace_efield_L1_z, alpha=0.5, label="polarization v")
        ax4.set_title(f"efield L1, antenna {du_idx}")
        ax4.set_xlabel("time in ns")
        ax4.set_ylabel("efield in uV/m")


        if t_0_shift == True:
          ax1.axvline(t0_voltage_L0[du_idx], label="t0 Trigger")
          ax2.axvline(t0_adc_L1[du_idx], label="t0 Trigger")
          ax3.axvline(t0_efield_L0[du_idx], label="t0 Trigger")
          ax4.axvline(t0_efield_L1[du_idx], label="t0 Trigger")
        else:
          ax1.axvline(0, label="t0 Trigger")
          ax2.axvline(0, label="t0 Trigger")
          ax3.axvline(0, label="t0 Trigger")
          ax4.axvline(0, label="t0 Trigger")

        # Add common vertical line (assuming same time axis)
        for ax in [ax1, ax2, ax3, ax4]:
          ax.legend(loc="upper right")

        plt.tight_layout()
        if(args.savefig):
<<<<<<< HEAD
          plt.savefig(f"{directory}/IllustrateSimPipe_{run_number}_{event_number}_{du_idx}_{savelabel}.png")
=======
          plt.savefig(f"{plot_dir}/plots/IllustrateSimPipe_{run_number}_{event_number}_{du_idx}_{savelabel}.png")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
          plt.close(fig)
        else: 
          plt.show()



def plot_time_map(directory, simulator=None):
  d_input = groot.DataDirectory(directory)

  #Get the trees L0
  trun_l0 = d_input.trun_l0
  tshower_l0 = d_input.tshower_l0
  tefield_l0 = d_input.tefield_l0
  tvoltage_l0 = d_input.tvoltage_l0           
  #Get the trees L1
  tefield_l1 = d_input.tefield_l1
  tadc_l1 = d_input.tadc_l1
  trun_l1 = d_input.trun_l1

  #get the list of events
  events_list = tefield_l1.get_list_of_events()
  nb_events = len(events_list)

  # If there are no events in the file, exit
  if nb_events == 0:
    sys.exit("There are no events in the file! Exiting.")
  
  ####################################################################################
  # start looping over the events
  ####################################################################################
  previous_run = None    
  for event_number,run_number in events_list:
      assert isinstance(event_number, int)
      assert isinstance(run_number, int)
      logger.info(f"Running event_number: {event_number}, run_number: {run_number}")
      
      tefield_l0.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
      tshower_l0.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
      tefield_l1.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
      tvoltage_l0.get_event(event_number, run_number)
      tadc_l1.get_event(event_number, run_number)
      
      #TODO: WRITE AN ISSUE. Ask Lech Would it be posible that get_run automatically does nothing if you ask for the run its currently being pointed to?
      if previous_run != run_number:                          # load only for new run.
        trun_l0.get_run(run_number)                         # update run info to get site latitude and longitude.
        trun_l1.get_run(run_number)                         # update run info to get site latitude and longitude.            
        previous_run = run_number
      
      
      trace_efield_L0 = np.asarray(tefield_l0.trace, dtype=np.float32)   # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
      du_id = np.asarray(tefield_l0.du_id) # MT: used for printing info and saving in voltage tree.
      # JK: du_id is currently unused - TODO


      # t0 calculations
      event_second = tshower_l0.core_time_s
      event_nano = tshower_l0.core_time_ns
      t0_efield_L1 = (tefield_l1.du_seconds-event_second)*1e9  - event_nano + tefield_l1.du_nanoseconds 
      t0_efield_L0 = (tefield_l0.du_seconds-event_second)*1e9  - event_nano + tefield_l0.du_nanoseconds
      t0_adc_L1 = (tadc_l1.du_seconds-event_second)*1e9  - event_nano + tadc_l1.du_nanoseconds
      t0_voltage_L0 = (tvoltage_l0.du_seconds-event_second)*1e9  - event_nano + tvoltage_l0.du_nanoseconds 

      #TODO: this forces a homogeneous antenna array.
      trace_shape = trace_efield_L0.shape  # (nb_du, 3, tbins of a trace)
      nb_du = trace_shape[0]
      sig_size = trace_shape[-1]
      logger.info(f"Event has {nb_du} DUs, with a signal size of: {sig_size}")
      
      #this gives the indices of the antennas of the array participating in this event
      event_dus_indices = tefield_l0.get_dus_indices_in_run(trun_l0)

      du_xyzs= np.asarray(trun_l0.du_xyz)[event_dus_indices] 
      # MT: antenna positions in shc? coordinates (TODO: Shouldnt this be in grand coordinates?)
      # JK: yes, everything should be in grand coordinates and conventions.
      
      zenith  = tshower_l0.zenith
      azimuth = tshower_l0.azimuth

      if simulator == re.match(r"^coreas$", simulator, flags=re.IGNORECASE):
        # use this re check so it's not case sensitive
        print("loading Coreas axis layout...")
        # rotate x and y by 90° -> x = North, y = West
        x, y = du_xyzs[:,1], du_xyzs[:,0]
      else:
        # keep x and y as is
        x, y = du_xyzs[:,0], du_xyzs[:,1]

      # Plot arrival time distribution
      # Create a figure with subplots to match the other plot 
      fig, axs = plt.subplots(2,2, figsize=(8, 6))
      plt.suptitle(f"arrival time distribution, event {event_number}, run {run_number},\n azimuth {azimuth}°, zenith {zenith}°")

      # Plot voltage traces on the first subplot
      ax1=axs[0,0]
      map = ax1.scatter(x, y, c=t0_voltage_L0, marker='o', s=20)
      ax1.set_title(f"voltage")
      cbar = plt.colorbar(map, ax=ax1)
      cbar.set_label("arrival time in ns")

      # adc traces
      ax2=axs[0,1]
      map = ax2.scatter(x, y, c=t0_adc_L1, marker='o', s=20)
      ax2.set_title(f"adc")
      cbar = plt.colorbar(map, ax=ax2)
      cbar.set_label("arrival time in ns")

      # Plot electric field L1
      ax3=axs[1,0]
      map = ax3.scatter(x, y, c=t0_efield_L0, marker='o', s=20)
      ax3.set_title(f"efield L0")
      cbar = plt.colorbar(map, ax=ax3)
      cbar.set_label("arrival time in ns")


      # Plot electric field L2
      ax4=axs[1,1]
      map = ax4.scatter(x, y, c=t0_efield_L1, marker='o', s=20)
      ax4.set_title(f"efield L1")
      cbar = plt.colorbar(map, ax=ax4)
      cbar.set_label("arrival time in ns")


      # Add common vertical line (assuming same time axis)
      if simulator == "Coreas":
        for ax in [ax1, ax2, ax3, ax4]:
          ax.set_xlabel("West in m")
          ax.set_ylabel("North in m")
          ax.axis('equal')
      else:
        for ax in [ax1, ax2, ax3, ax4]:
          ax.set_xlabel("antenna position X in m")
          ax.set_ylabel("antenna position Y in m")
          ax.axis('equal')

      plt.tight_layout()
      if(args.savefig):
<<<<<<< HEAD
        plt.savefig(f"{directory}/TimeMap_{run_number}_{event_number}.png")
=======
        plt.savefig(f"{plot_dir}/plots/TimeMap_{run_number}_{event_number}.png")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
        plt.close(fig)
      else: 
         plt.show()
    


def get_tree_du_id_and_xyz(trawefield,shower_core):
    # *** Store the DU's to run - they needed to be collected from all events ***
    # Get the ids and positions from all the events
    count = trawefield.draw("du_id:du_x:du_y:du_z", "", "goff")
    du_ids = np.array(np.frombuffer(trawefield.get_v1(), dtype=np.float64, count=count)).astype(int)
    du_xs = np.array(np.frombuffer(trawefield.get_v2(), dtype=np.float64, count=count)).astype(np.float32)
    du_ys = np.array(np.frombuffer(trawefield.get_v3(), dtype=np.float64, count=count)).astype(np.float32)
    du_zs = np.array(np.frombuffer(trawefield.get_v4(), dtype=np.float64, count=count)).astype(np.float32)
    

<<<<<<< HEAD
    #trawefield has the antenna positions in shc, and we need to put them in array coordinates, or this is all messed up.
    #TODO: This is a flat earth approximation that asumes shower core is in xc,yc,zc of flat cordinate system centered in the array, origin at ground.
    # and that du_xs are given in a flat shower coordinate system , with origin in the core, but at sea level (in this system the core coordinates are 0,0,groundaltitude)
    print("Warning: using flat earth approximation for coordinates!. Core:",shower_core)
    du_xs = du_xs + shower_core[0]
    du_ys = du_ys + shower_core[1]
    du_zs = du_zs

=======
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
    # Get indices of the unique du_ids
    # ToDo: sort?
    unique_dus_idx = np.unique(du_ids, return_index=True)[1]
    # Leave only the unique du_ids
    du_ids = du_ids[unique_dus_idx]
    # Stack x/y/z together and leave only the ones for unique du_ids
    du_xyzs = np.column_stack([du_xs, du_ys, du_zs])[unique_dus_idx]

    return np.asarray(du_ids), np.asarray(du_xyzs)    
    


def plot_raws(directory):
  # this glob is not perfect yet, but it's good enough for now
  # TODO: glob by EventID or something
  files = glob.glob(f"{directory}/../*.rawroot")
  if len(files) == 1:
    filename = files[0]
    print(f"Found rawroot file {filename}")
  else:
    print(f"Found rawroot files {files}")
    sys.exit("Please make sure there's only one .rawroot file in the directory above the one you specified.")

  trawefield = RawTrees.RawEfieldTree(filename)
  trawshower = RawTrees.RawShowerTree(filename)
  nentries = trawefield.get_entries()
  for i in range(nentries):
    trawefield.get_entry(i)
    du_ids, du_xyzs = get_tree_du_id_and_xyz(trawefield,trawshower.shower_core_pos)
    # this counting method is terrible, but somehow trawefield.trace_x[du] won't work
    count = 0
    for du in du_ids:
      print(f"Plotting raw trace of antenna {du}")
      trace_x = trawefield.trace_x[count]
      trace_y = trawefield.trace_y[count]
      trace_z = trawefield.trace_z[count]

      plt.title(f"raw trace {du}")
      plt.plot(trace_x, label="x")
      plt.plot(trace_y, label="y")
      plt.plot(trace_z, label="z")
      plt.xlabel("time in bins")
      plt.ylabel("efield in uV/m")
      plt.legend()
      
      plt.tight_layout()
      if(args.savefig):
<<<<<<< HEAD
         plt.savefig(f"{directory}/rawtrace_{du}.png")
=======
         plt.savefig(f"{plot_dir}/plots/rawtrace_{du}.png")
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
         plt.close()
      else: 
         plt.show()

      count += 1



if __name__ == "__main__":
  args = manage_args()
  mlg.create_output_for_logger(args.verbose, log_stdout=True)
  logger.info(mlg.string_begin_script())
  logger.info("Creating event plots in source directory "+args.directory)

<<<<<<< HEAD
  directory = args.directory
=======
  # change plot directory if specified
  if args.savefig:
    if args.savefig_dir is None:
      # Use simulation output directory by default
      plot_dir = args.directory
    else:
      # Use specified savefig_dir
      plot_dir = args.savefig_dir
  else:
    pass

  directory = args.directory
  savefig = args.savefig
  savefig_dir = args.savefig_dir

  if savefig:
    if savefig_dir is None:
      # Use simulation output directory by default
      plot_dir = directory
    else:
      # Use specified savefig_dir
      plot_dir = savefig_dir
  else:
    pass

  
>>>>>>> 543186835d8777d8ec9939efb94fd40457105c1f
  plot_core_positions(directory)
  plot_time_map(directory, simulator=args.sim)
  plot_traces_all_levels(directory, t_0_shift=False)
  #plot_raws(directory)

