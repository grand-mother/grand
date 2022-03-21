#!/usr/bin/python
# An example of storing traces to a file
import numpy as np
import time
from grand.io.root_trees import *

# Generate random number of traces with random lengths for 10 events, as can be in the real case
event_count = 10
adc_traces = []
traces = []
for ev in range(event_count):
    trace_count = np.random.randint(3,7)
    adc_traces.append([])
    traces.append([])
    for i in range(trace_count):
        # The trace length
        l = np.random.randint(900,1000)
        # X,Y,Z needed for each trace
        adc_traces[-1].append((np.random.randint(-20,21, l).astype(np.int16), np.random.randint(-20,21, l).astype(np.int16), np.random.randint(-20,21, l).astype(np.int16),np.random.randint(-20,21, l).astype(np.int16)))
        traces[-1].append(((adc_traces[-1][i][0]*0.9/8192).astype(np.float32), (adc_traces[-1][i][1]*0.9/8192).astype(np.float32), (adc_traces[-1][i][2]*0.9/8192).astype(np.float32)))

# ********** Generarte Run Tree ****************
# It needs to be first, so that the Event trees can find it. However, it need some informations from them, so will be filled at the end
trun = RunTree()
trun.first_event = 0
trun.last_event = event_count
trun.fill()
trun.write("stored_data.root")
print("Wrote trun")

# ********** ADC Counts ****************

# Create the ADC counts tree
tadccounts = ADCEventTree()

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
    # Triggered event
    tadccounts.event_type = 0x8000
    # The number of antennas in the event
    tadccounts.du_count = len(traces[ev])

    # Loop through the event's traces
    du_id = []
    du_seconds = []
    du_nanoseconds = []
    trigger_position = []
    trigger_flag = []
    atm_temperature = []
    atm_pressure = []
    atm_humidity = []
    acceleration_x = []
    acceleration_y = []
    acceleration_z = []
    trace_0 = []
    trace_1 = []
    trace_2 = []
    trace_3 = []
    for i,trace in enumerate(adc_traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        du_id.append(i)
        du_seconds.append(tadccounts.time_seconds)
        du_nanoseconds.append(tadccounts.time_nanoseconds)
        trigger_position.append(i//2)
        trigger_flag.append(tadccounts.event_type)
        atm_temperature.append(20+ev//2)
        atm_pressure.append(1024+ev//2)
        atm_humidity.append(50+ev//2)
        acceleration_x.append(ev//2)
        acceleration_y.append(ev//3)
        acceleration_z.append(ev//4)

        trace_0.append(trace[0]+1)
        trace_1.append(trace[1]+2)
        trace_2.append(trace[2]+3)
        trace_3.append(trace[3]+4)

    tadccounts.du_id = du_id
    tadccounts.du_seconds = du_seconds
    tadccounts.du_nanoseconds = du_nanoseconds
    tadccounts.trigger_position = trigger_position
    tadccounts.trigger_flag = trigger_flag
    tadccounts.atm_temperature = atm_temperature
    tadccounts.atm_pressure = atm_pressure
    tadccounts.atm_humidity = atm_humidity
    tadccounts.acceleration_x = acceleration_x
    tadccounts.acceleration_y = acceleration_y
    tadccounts.acceleration_z = acceleration_z
    tadccounts.trace_0 = trace_0
    tadccounts.trace_1 = trace_1
    tadccounts.trace_2 = trace_2
    tadccounts.trace_3 = trace_3

    tadccounts.fill()

# write the tree to the storage
tadccounts.write("stored_data.root")
print("Wrote tadccounts")

# ********** Voltage ****************

# Voltage has the same data as ADC counts tree, but recalculated to "real" (usually float) values

# Recalculate ADC counts to voltage, just with a dummy conversion now: 0.9 V is equal to 8192 counts for XiHu data
adc2v = 0.9/8192

# Create the ADC counts tree
tvoltage = VoltageEventTree()

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
    # Triggered event
    tvoltage.event_type = 0x8000
    # The number of antennas in the event
    tvoltage.du_count = len(traces[ev])

    # Loop through the event's traces
    du_id = []
    du_seconds = []
    du_nanoseconds = []
    trigger_position = []
    trigger_flag = []
    atm_temperature = []
    atm_pressure = []
    atm_humidity = []
    acceleration_x = []
    acceleration_y = []
    acceleration_z = []
    trace_x = []
    trace_y = []
    trace_z = []
    for i,trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        du_id.append(i)
        du_seconds.append(tvoltage.time_seconds)
        du_nanoseconds.append(tvoltage.time_nanoseconds)
        trigger_position.append(i//2)
        trigger_flag.append(tvoltage.event_type)
        atm_temperature.append(20+ev/2)
        atm_pressure.append(1024+ev/2)
        atm_humidity.append(50+ev/2)
        acceleration_x.append(ev/2)
        acceleration_y.append(ev/3)
        acceleration_z.append(ev/4)

        trace_x.append(trace[0])
        # print(trace_x[-1], type(trace_x[-1]))
        # exit()
        trace_y.append(trace[1])
        trace_x.append(trace[2])

    tvoltage.du_id = du_id
    tvoltage.du_seconds = du_seconds
    tvoltage.du_nanoseconds = du_nanoseconds
    tvoltage.trigger_position = trigger_position
    tvoltage.trigger_flag = trigger_flag
    tvoltage.atm_temperature = atm_temperature
    tvoltage.atm_pressure = atm_pressure
    tvoltage.atm_humidity = atm_humidity
    tvoltage.acceleration_x = acceleration_x
    tvoltage.acceleration_y = acceleration_y
    tvoltage.acceleration_z = acceleration_z
    tvoltage.trace_x = trace_x
    tvoltage.trace_y = trace_y
    tvoltage.trace_z = trace_z

    tvoltage.fill()

# write the tree to the storage
tvoltage.write("stored_data.root")
print("Wrote tvoltage")

# ********** Efield ****************

# Efield has the same data as ADC counts tree and Voltage tree + FFTs
from scipy import fftpack

# Recalculate Voltage to Efield - just an example, so just multiply by a dumb value
# Here the GRANDlib Efield computation function with antenna model should be used
v2ef = 1.17

# Create the ADC counts tree
tefield = EfieldEventTree()

# fill the tree with every second of generated events - dumb selection
for ev in range(0,event_count,2):
    tefield.event_number = ev
    # Loop through the event's traces
    traces_lengths = []
    start_times = []
    rel_peak_times = []
    det_times = []
    e_det_times = []
    isTriggereds = []
    sampling_speeds = []
    trace_xs = []
    trace_ys = []
    trace_zs = []
    fft_mag_xs = []
    fft_mag_ys = []
    fft_mag_zs = []
    fft_phase_xs = []
    fft_phase_ys = []
    fft_phase_zs = []

    for i,trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        traces_lengths.append(len(trace[0]))
        start_times.append(ev*10)
        rel_peak_times.append(ev*11)
        det_times.append(ev*13.5)
        e_det_times.append(ev*14)
        isTriggereds.append(True)
        sampling_speeds.append(ev*15)

        # To multiply a list by a number elementwise, convert to a numpy array and back
        # Here a real ComputeEfield() function should be called instead of multiplying adc2v
        # ToDo: better read the Voltage trace from the TTree
        trace_xs.append((np.array(trace[0])*adc2v*v2ef).tolist())
        trace_ys.append((np.array(trace[1])*adc2v*v2ef).tolist())
        trace_zs.append((np.array(trace[2])*adc2v*v2ef).tolist())

        # FFTS
        fft = fftpack.fft(trace[0])
        fft_mag_xs.append(np.abs(fft))
        # ToDo: recall how to calculate the phase easily
        fft_phase_xs.append(np.abs(fft))
        fft = fftpack.fft(trace[1])
        fft_mag_ys.append(np.abs(fft))
        # ToDo: recall how to calculate the phase easily
        fft_phase_ys.append(np.abs(fft))
        fft = fftpack.fft(trace[2])
        fft_mag_zs.append(np.abs(fft))
        # ToDo: recall how to calculate the phase easily
        fft_phase_zs.append(np.abs(fft))

    tefield.det_id = list(range(len(traces[ev])))
    tefield.trace_length = traces_lengths
    tefield.start_time = start_times
    tefield.rel_peak_time = rel_peak_times
    tefield.det_time = det_times
    tefield.e_det_time = e_det_times
    tefield.isTriggered = isTriggereds
    tefield.sampling_speed = sampling_speeds
    tefield.trace_x = trace_xs
    tefield.trace_y = trace_ys
    tefield.trace_z = trace_zs
    tefield.fft_mag_x = fft_mag_xs
    tefield.fft_mag_y = fft_mag_ys
    tefield.fft_mag_z = fft_mag_zs
    tefield.fft_phase_x = fft_phase_xs
    tefield.fft_phase_y = fft_phase_ys
    tefield.fft_phase_z = fft_phase_zs

    tefield.fill()

# write the tree to the storage
tefield.write("stored_data.root")
print("Wrote tefield")

# tree_file.Close()
