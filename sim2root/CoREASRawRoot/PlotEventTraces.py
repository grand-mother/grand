#!/usr/bin/python
# An example of using DataFile for ROOT file reading
import numpy as np
import sys
import os
from grand.dataio.root_trees import *
from types import SimpleNamespace
import matplotlib.pyplot as plt



# Need to provide a file to read
if len(sys.argv)<2:
    print("Please provide a tefield_*.root file name to read")
    exit()



f_input = sys.argv[1]       

directory_path, fname = os.path.split(f_input)

f_input_TRun=directory_path+"/trun_"+ fname[8:]              
f_input_TShower=directory_path+"/tshower_"+ fname[8:]
f_input_TRunShowerSim=directory_path+"/trunshowersim_"+ fname[8:]         
f_input_TShowerSim=directory_path+"/tshowersim_"+ fname[8:]
f_input_TRunEfieldSim=directory_path+"/trunefieldsim_"+ fname[8:] 
f_input_TEfield=directory_path+"/tefield_"+ fname[8:] 

f_input_TVoltage=directory_path+"/tvoltage_"+ fname[8:-5] +"_with-rf_with-noise.root"
f_input_TVoltageNN=directory_path+"/tvoltage_"+ fname[8:-5] +"_with-rf_no-noise.root"
f_input_TVoltageNRF=directory_path+"/tvoltage_"+ fname[8:-5] +"_no-rf_with-noise.root"
f_input_TVoltageNRFNN=directory_path+"/tvoltage_"+ fname[8:-5] +"_no-rf_no-noise.root"
  
gt = SimpleNamespace()
gt.trun = TRun(f_input_TRun)
gt.trun.get_entry(0)
gt.trunshowersim = TRunShowerSim(f_input_TRunShowerSim)
gt.trunshowersim.get_entry(0)
gt.trunefieldsim = TRunEfieldSim(f_input_TRunEfieldSim)
gt.trunefieldsim.get_entry(0)
gt.tshower = TShower(f_input_TShower)
gt.tshower.get_entry(0)
gt.tshowersim = TShowerSim(f_input_TShowerSim)
gt.tshowersim.get_entry(0)
gt.tefield = TEfield(f_input_TEfield)
gt.tefield.get_entry(0)
gt.tvoltage = TVoltage(f_input_TVoltage)
gt.tvoltage.get_entry(0)
gt.tvoltageNN = TVoltage(f_input_TVoltageNN)
gt.tvoltageNN.get_entry(0)
gt.tvoltageNRF = TVoltage(f_input_TVoltageNRF)
gt.tvoltageNRF.get_entry(0)
gt.tvoltageNRFNN = TVoltage(f_input_TVoltageNRFNN)
gt.tvoltageNRFNN.get_entry(0)

timeseconds=gt.tefield.time_seconds
timenanoseconds=gt.tefield.time_nanoseconds

#change all times relative to the first time TODO:(could be done to the event time, but this first events have time set to 0)
du_nanoseconds=np.array(gt.tefield.du_nanoseconds)
du_seconds=np.array(gt.tefield.du_seconds)
du_nanoseconds=du_nanoseconds-du_nanoseconds
du_seconds=du_seconds-du_seconds[0]
du_nanoseconds=du_nanoseconds+du_seconds*1e9

etbinsize=float(gt.trun.t_bin_size[0]) #TODO this should be in trunefieldsim
etracelenght=len(gt.tefield.trace[0][0])


vtbinsize=2                            #sorry this is hardwiredfor now
vtracelenght=len(gt.tvoltage.trace[0][0])

print(etracelenght,vtracelenght)

tpre=gt.trunefieldsim.t_pre
tpost=gt.trunefieldsim.t_post

etimebins=np.arange(0,etracelenght*etbinsize,etbinsize)
vtimebins=np.arange(0,vtracelenght*vtbinsize,vtbinsize)

print(len(etimebins),len(vtimebins))
# Number of antennas
num_antennas = int(gt.tefield.du_count)
# Number of traces per figure
traces_per_figure = 2


# Loop through traces and create figures
for start_index in range(0, num_antennas, traces_per_figure):
    end_index = start_index + traces_per_figure
    # Ensure the end_index does not exceed the total number of antennas
    end_index = min(end_index, num_antennas)
    
    # Create a figure
    fig, axs = plt.subplots(traces_per_figure, 2, sharex=True, sharey=True, figsize=(12, 8))

    # Loop through traces for the current figure
    for i, ax_row in enumerate(axs):
        antenna_index = start_index + i  
        
        if antenna_index < num_antennas:
            #time=timebins+du_nanoseconds[antenna_index]-tpre 
            print(antenna_index)
            time=etimebins
            axs[i][0].plot(time,np.array(gt.tefield.trace[antenna_index][0])*10, label='Efield X x10',color="orange")
            time=vtimebins
            axs[i][0].scatter(time,np.array(gt.tvoltage.trace[antenna_index][0]),label='$V_{RF}$ X + noise',color="red",s=2)
            axs[i][0].plot(time,np.array(gt.tvoltageNN.trace[antenna_index][0]),label='$V_{RF}$ X',color="blue")
            axs[i][0].scatter(time,np.array(gt.tvoltageNRF.trace[antenna_index][0]),label='$V_{OC}$ Y + noise',color="green",s=2) 
            axs[i][0].plot(time,np.array(gt.tvoltageNRFNN.trace[antenna_index][0]),label='$V_{OC}$ X',color="black")                      
            axs[i][0].set_ylabel('Amplitude [uV/m] or [uV]')
            
    
            time=etimebins
            axs[i][1].plot(time,np.array(gt.tefield.trace[antenna_index][1])*10, label='Efield Y x10',color="orange")
            time=vtimebins
            axs[i][1].scatter(time,np.array(gt.tvoltage.trace[antenna_index][1]),label='$V_{RF}$ Y + noise',color="red",s=2)
            axs[i][1].plot(time,np.array(gt.tvoltageNN.trace[antenna_index][1]),label='$V_{RF}$ Y',color="blue")
            axs[i][1].scatter(time,np.array(gt.tvoltageNRF.trace[antenna_index][1]),label='$V_{OC}$ Y + noise',color="green",s=2)
            axs[i][1].plot(time,np.array(gt.tvoltageNRFNN.trace[antenna_index][1]),label='$V_{OC}$ Y',color="black")            

            #axs[i][1].set_ylabel('Amplitude')    

            # Add legend and labels
            axs[-1][0].set_xlabel('Time [ns]')
            axs[-1][1].set_xlabel('Time [ns]')
            #axs[-1][0].set_yscale("symlog", linthresh=5e2)
            #axs[-1][1].set_yscale("symlog", linthresh=5e2)
            
            axs[0][0].legend()
            axs[0][1].legend()

# Adjust layout
    plt.tight_layout()

plt.show()


