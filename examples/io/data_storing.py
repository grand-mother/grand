#!/usr/bin/python

'''
An example of storing traces to a file using root_trees.py
'''
import numpy as np
import time
import sys
from grand.io.root_trees import *

# Check if a filename was provided on the command line
if len(sys.argv) == 2 and sys.argv[1][-5:] == ".root":
    filename = sys.argv[1]
else:
    filename = "stored_data.root"

# Generate random number of traces with random lengths for 10 events, 
# as can be in the real case
event_count = 10
total_du = 96                   # 8*12 antennas
x = np.arange(-2000,2000, 500)  # 8 antennas
y = np.arange(-3000,3000, 500)  # 12 antennas
X,Y = np.meshgrid(x,y)
x = X.flatten()
y = Y.flatten()
z = np.random.randint(1000,1400, len(x))
du_xyz = np.column_stack((x,y,z))
# Recalculate ADC counts to voltage, just with a dummy conversion now: 
# 0.9 V is equal to 8192 counts for XiHu data
adc2v = 0.9 / 8192

# Generate traces now to be used later.
adc_traces = []   # final shape (events, du_count, 4, trace_length)
traces = []       # final shape (events, du_count, 3, trace_length)
for ev in range(event_count):
    du_count = np.random.randint(3, 7) # number of antenna hit in each event.
    l = np.random.randint(900, 1000)   # length of time traces
    adc_traces.append(np.random.randint(-20, 21, (du_count, 4, l))) # 4 channels
    traces.append(adc_traces[ev][:,:3,:] * adc2v)  # only use 0,1,2 channels.

# ********** Generate Run Tree ****************
# It needs to be first, so that the Event trees can find it. 
# However, it need some informations from them, so will be filled at the end
trun = TRun()
trun.run_number = 0
trun.site = "dummy site"
trun.first_event = 0
trun.last_event = event_count
trun.site = "Dunhuang"
trun.site_layout = "Grid"
trun.origin_geoid = np.array([40.98, 93.93, 1287.0]) # [latitude, longitude, height]
trun.du_id = np.arange(96) # 0,1,2,...95
trun.du_xyz= du_xyz
trun.t_bin_size= np.array([0.5])
trun.fill()
trun.write(filename)
print("Wrote trun")

# ********** ADC Counts ****************
# Create the ADC counts tree
tadccounts = TADC()

# fill the tree with the generated events
for ev in range(event_count):
    tadccounts.run_number = 0
    tadccounts.event_number = ev
    # First data unit in the event
    tadccounts.first_du = 0
    # As the event time add the current time
    tadccounts.time_seconds = int(time.mktime(time.gmtime()))
    # Event nanoseconds 0 for now
    tadccounts.time_nanoseconds = 0
    # Trigger type 0x1000 10 s trigger and 0x8000 random trigger, else shower
    tadccounts.event_type = 0x8000
    # The number of antennas in the event
    tadccounts.du_count = adc_traces[ev].shape[0]
    tadccounts.du_id = np.random.randint(0,96,tadccounts.du_count)
    tadccounts.du_seconds = tadccounts.time_seconds+np.arange(tadccounts.du_count)
    tadccounts.du_nanoseconds = np.zeros(tadccounts.du_count, dtype=int)
    # Trigger position in the trace (trigger start = nanoseconds - 2*sample number)
    tadccounts.trigger_position = np.arange(tadccounts.du_count)//2
    tadccounts.trigger_flag = np.repeat(tadccounts.event_type, tadccounts.du_count)
    tadccounts.atm_temperature = np.repeat(20 + ev // 2, tadccounts.du_count)
    tadccounts.atm_pressure = np.repeat(1024 + ev // 2, tadccounts.du_count)
    tadccounts.atm_humidity = np.repeat(50 + ev // 2, tadccounts.du_count)
    tadccounts.acceleration_x = np.repeat(ev // 2, tadccounts.du_count)
    tadccounts.acceleration_y = np.repeat(ev // 3, tadccounts.du_count)
    tadccounts.acceleration_z = np.repeat(ev // 4, tadccounts.du_count)
    tadccounts.trace_ch = adc_traces[ev]
    tadccounts.fill()

# write the tree to the storage
tadccounts.write(filename)
print("Wrote tadccounts")

# ********** Voltage ****************
# Voltage has the same data as ADC counts tree, but recalculated to "real" (usually float) values

# Create the ADC counts tree
tvoltage = TVoltage()

# fill the tree with the generated events
for ev in range(event_count):
    tvoltage.run_number = 0
    tvoltage.event_number = ev
    # First data unit in the event
    tvoltage.first_du = 0
    # As the event time add the current time
    tvoltage.time_seconds = int(time.mktime(time.gmtime()))
    # Event nanoseconds 0 for now
    tvoltage.time_nanoseconds = 0
    tvoltage.du_count = traces[ev].shape[0]
    tvoltage.du_id = np.random.randint(0,96,tvoltage.du_count)
    tvoltage.du_seconds = tvoltage.time_seconds+np.arange(tvoltage.du_count)
    tvoltage.du_nanoseconds = np.zeros(tvoltage.du_count, dtype=int)
    tvoltage.du_acceleration = np.random.rand(tvoltage.du_count, 3)
    tvoltage.trace = traces[ev]

    tvoltage.fill()

# write the tree to the storage
tvoltage.write(filename)
print("Wrote tvoltage")

# ********** Efield ****************

# Efield has some of the Voltage tree data + FFTs
from scipy import fftpack

# Recalculate Voltage to Efield - just an example, so just multiply by a dumb value
# Here the GRANDlib Efield computation function with antenna model should be used
v2ef = 1.17

# Create the ADC counts tree
tefield = TEfield()

# fill the tree with every second of generated events - dumb selection
for ev in range(0, event_count, 2):
    tefield.run_number = 0
    tefield.event_number = ev
    # Unix time corresponding to the GPS seconds of the trigger
    tefield.time_seconds = int(time.mktime(time.gmtime()))
    # GPS nanoseconds corresponding to the trigger of the first triggered station
    # Event nanoseconds 0 for now
    tefield.time_nanoseconds = 0
    # Triggered event
    tefield.event_type = 0x8000
    # The number of antennas in the event
    tefield.du_count = traces[ev].shape[0]
    tefield.du_id = np.random.randint(0,96,tefield.du_count)
    tefield.du_seconds = tefield.time_seconds+np.arange(tefield.du_count)
    tefield.du_nanoseconds = np.zeros(tefield.du_count, dtype=int)
    tefield.trace = traces[ev] * v2ef
    tefield.fft_mag = np.abs(fftpack.fft(traces[ev]))
    tefield.fft_phase = np.angle(fftpack.fft(traces[ev]), deg=True)

    tefield.fill()

# write the tree to the storage, but don't close the file - it will be used for tshower
# ToDo: is this correct? Not sure if I should use the file opened for writing when I am reading
tefield.write(filename, close_file=False)
print("Wrote tefield")

# Generation of shower data for each event - this should be reonstruction, but here just dumb values
tshower = TShower()
# Loop through all Efield entries
for i in range(tefield.get_entries()):
    # Get the Efield event
    tefield.get_entry(i)

    tshower.run_number = tefield.run_number
    tshower.event_number = tefield.event_number

    tshower.primary_type = "particle"
    tshower.energy_primary = np.random.random(1) * 1e8
    tshower.azimuth = np.random.random(1) * 360
    tshower.zenith = np.random.random(1) * 180 - 90
    tshower.shower_core_pos = np.random.random(3)
    tshower.atmos_model = "dense air dummy"
    tshower.atmos_model_param = np.random.random(3)
    tshower.magnetic_field = np.random.random(3)
    tshower.core_alt = 3000.0 + np.random.randint(0, 1000)
    tshower.xmax_grams = np.random.random(1) * 500
    tshower.xmax_pos = np.random.random(3) # Xmax in GRANDCS.
    tshower.xmax_pos_shc = np.random.random(3) # Xmax in shower coordinates.
    tshower.core_time_s = np.random.randint(0, 10000) * 1.0
    tshower.core_time_ns = 0

    tshower.fill()

tshower.write(filename)
print("Wrote tshower")

print(f"Finished writing file {filename}")
