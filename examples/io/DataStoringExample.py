#!/usr/bin/python
# An example of storing traces to a file
import numpy as np
# This does not work now...
#from grand.io.root_trees import *
import sys
sys.path.append("../../grand/io")
from root_trees import *
import ROOT

# Generate random number of traces with random lengths for 10 events, as can be in the real case
event_count = 10
traces = []
for ev in range(event_count):
    trace_count = np.random.randint(3,7)
    traces.append([])
    for i in range(trace_count):
        # The trace length
        l = np.random.randint(900,1000)
        # X,Y,Z needed for each trace
        traces[-1].append((np.random.randint(-20,21, l).astype(np.float32), np.random.randint(-20,21, l).astype(np.float32), np.random.randint(-20,21, l).astype(np.float32)))

# ********** Generarte Run Tree ****************
# It needs to be first, so that the Event trees can find it. However, it need some informations from them, so will be filled at the end
trun = RunTree()
trun.first_event = 0
trun.last_event = event_count
trun.fill()
trun.write("stored_data.root")

# ********** ADC Counts ****************

# Create the ADC counts tree
tadccounts = ADCEventTree()

# fill the tree with the generated events
for ev in range(event_count):
    tadccounts.event_number = ev
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
    for i,trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        traces_lengths.append(len(trace[0]))
        start_times.append(ev*10)
        rel_peak_times.append(ev*11)
        det_times.append(ev*13)
        e_det_times.append(ev*14)
        isTriggereds.append(True)
        sampling_speeds.append(ev*15)

        trace_xs.append(trace[0])
        trace_ys.append(trace[1])
        trace_zs.append(trace[2])

    tadccounts.det_id = list(range(len(traces[ev])))
    tadccounts.trace_length = traces_lengths
    tadccounts.start_time = start_times
    tadccounts.rel_peak_time = rel_peak_times
    tadccounts.det_time = det_times
    tadccounts.e_det_time = e_det_times
    tadccounts.isTriggered = isTriggereds
    tadccounts.sampling_speed = sampling_speeds
    tadccounts.trace_0 = trace_xs
    tadccounts.trace_1 = trace_ys
    tadccounts.trace_2 = trace_zs

    tadccounts.fill()

# write the tree to the storage
tadccounts.write("stored_data.root")
print("Wrote tadccounts")

# ********** Voltage ****************

# Voltage has the same data as ADC counts tree, but the ADC counts are recalculated to voltage

# Recalculate ADC counts to voltage, just with a dummy conversion now: 0.9 V is equal to 8192 counts for XiHu data
adc2v = 0.9/8192

# Create the ADC counts tree
tvoltage = VoltageEventTree()

# fill the tree with the generated events
for ev in range(event_count):
    tvoltage.event_number = ev
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
    for i,trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        traces_lengths.append(len(trace[0]))
        start_times.append(ev*10)
        rel_peak_times.append(ev*11)
        det_times.append(ev*13.2)
        e_det_times.append(ev*14)
        isTriggereds.append(True)
        sampling_speeds.append(ev*15)

        # To multiply a list by a number elementwise, convert to a numpy array and back
        trace_xs.append((np.array(trace[0])*adc2v).tolist())
        trace_ys.append((np.array(trace[1])*adc2v).tolist())
        trace_zs.append((np.array(trace[2])*adc2v).tolist())

    tvoltage.det_id = list(range(len(traces[ev])))
    tvoltage.trace_length = traces_lengths
    tvoltage.start_time = start_times
    tvoltage.rel_peak_time = rel_peak_times
    tvoltage.det_time = det_times
    tvoltage.e_det_time = e_det_times
    tvoltage.isTriggered = isTriggereds
    tvoltage.sampling_speed = sampling_speeds
    tvoltage.trace_x = trace_xs
    tvoltage.trace_y = trace_ys
    tvoltage.trace_z = trace_zs

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
