#!/usr/bin/python
# An example of storing traces to a file
import numpy as np
import time
import sys
from grand.io.root_trees import *

# Check if a filename was provided on the command line
if len(sys.argv) == 2 and sys.argv[1][-5:] == ".root":
    filename = sys.argv[1]
else:
    filename = "stored_data.root"

# Generate random number of traces with random lengths for 10 events, as can be in the real case
event_count = 10
adc_traces = []
traces = []
for ev in range(event_count):
    trace_count = np.random.randint(3, 7)
    adc_traces.append([])
    traces.append([])
    for i in range(trace_count):
        # The trace length
        l = np.random.randint(900, 1000)
        # X,Y,Z needed for each trace
        adc_traces[-1].append(
            (
                np.random.randint(-20, 21, l).astype(np.int16),
                np.random.randint(-20, 21, l).astype(np.int16),
                np.random.randint(-20, 21, l).astype(np.int16),
                np.random.randint(-20, 21, l).astype(np.int16),
            )
        )
        traces[-1].append(
            (
                (adc_traces[-1][i][0] * 0.9 / 8192).astype(np.float32),
                (adc_traces[-1][i][1] * 0.9 / 8192).astype(np.float32),
                (adc_traces[-1][i][2] * 0.9 / 8192).astype(np.float32),
            )
        )

# ********** Generarte Run Tree ****************
# It needs to be first, so that the Event trees can find it. However, it need some informations from them, so will be filled at the end
trun = RunTree()
trun.comment = "Generated DataStoringExample.py"
trun.run_number = 0
trun.site = "dummy site"
trun.first_event = 0
trun.last_event = event_count
trun.fill()
trun.write(filename)
print("Wrote trun")

# ********** ADC Counts ****************

# Create the ADC counts tree
tadccounts = ADCEventTree()
tadccounts.comment = "Generated DataStoringExample.py"

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
    for i, trace in enumerate(adc_traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        du_id.append(i)
        du_seconds.append(tadccounts.time_seconds)
        du_nanoseconds.append(tadccounts.time_nanoseconds)
        trigger_position.append(i // 2)
        trigger_flag.append(tadccounts.event_type)
        atm_temperature.append(20 + ev // 2)
        atm_pressure.append(1024 + ev // 2)
        atm_humidity.append(50 + ev // 2)
        acceleration_x.append(ev // 2)
        acceleration_y.append(ev // 3)
        acceleration_z.append(ev // 4)

        trace_0.append(trace[0] + 1)
        trace_1.append(trace[1] + 2)
        trace_2.append(trace[2] + 3)
        trace_3.append(trace[3] + 4)

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
tadccounts.write(filename)
print("Wrote tadccounts")

# ********** Voltage ****************

# Voltage has the same data as ADC counts tree, but recalculated to "real" (usually float) values

# Recalculate ADC counts to voltage, just with a dummy conversion now: 0.9 V is equal to 8192 counts for XiHu data
adc2v = 0.9 / 8192

# Create the ADC counts tree
tvoltage = VoltageEventTree()
tvoltage.comment = "Generated DataStoringExample.py"

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
    for i, trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        du_id.append(i)
        du_seconds.append(tvoltage.time_seconds)
        du_nanoseconds.append(tvoltage.time_nanoseconds)
        trigger_position.append(i // 2)
        trigger_flag.append(tvoltage.event_type)
        atm_temperature.append(20 + ev / 2)
        atm_pressure.append(1024 + ev / 2)
        atm_humidity.append(50 + ev / 2)
        acceleration_x.append(ev / 2)
        acceleration_y.append(ev / 3)
        acceleration_z.append(ev / 4)

        trace_x.append(trace[0])
        trace_y.append(trace[1])
        trace_z.append(trace[2])

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
tvoltage.write(filename)
print("Wrote tvoltage")

# ********** Efield ****************

# Efield has some of the Voltage tree data + FFTs
from scipy import fftpack

# Recalculate Voltage to Efield - just an example, so just multiply by a dumb value
# Here the GRANDlib Efield computation function with antenna model should be used
v2ef = 1.17

# Create the ADC counts tree
tefield = EfieldEventTree()
tefield.comment = "Generated DataStoringExample.py"

# fill the tree with every second of generated events - dumb selection
for ev in range(0, event_count, 2):
    tefield.run_number = 0
    tefield.event_number = ev
    # First data unit in the event
    tefield.first_du = 0
    # As the event time add the current time
    tefield.time_seconds = int(time.mktime(time.gmtime()))
    # Event nanoseconds 0 for now
    tefield.time_nanoseconds = 0
    # Triggered event
    tefield.event_type = 0x8000
    # The number of antennas in the event
    tefield.du_count = len(traces[ev])

    # Loop through the event's traces
    du_id = []
    du_seconds = []
    du_nanoseconds = []
    trigger_position = []
    trigger_flag = []
    atm_temperature = []
    atm_pressure = []
    atm_humidity = []
    trace_xs = []
    trace_ys = []
    trace_zs = []
    fft_mag_xs = []
    fft_mag_ys = []
    fft_mag_zs = []
    fft_phase_xs = []
    fft_phase_ys = []
    fft_phase_zs = []

    for i, trace in enumerate(traces[ev]):
        # print(ev,i, len(trace[0]))

        # Dumb values just for filling
        du_id.append(i)
        du_seconds.append(tefield.time_seconds)
        du_nanoseconds.append(tefield.time_nanoseconds)
        trigger_position.append(i // 2)
        trigger_flag.append(tefield.event_type)
        atm_temperature.append(20 + ev / 2)
        atm_pressure.append(1024 + ev / 2)
        atm_humidity.append(50 + ev / 2)

        # To multiply a list by a number elementwise, convert to a numpy array and back
        # Here a real ComputeEfield() function should be called instead of multiplying adc2v
        # ToDo: better read the Voltage trace from the TTree
        trace_xs.append((np.array(trace[0]) * v2ef).astype(np.float32).tolist())
        trace_ys.append((np.array(trace[1]) * v2ef).astype(np.float32).tolist())
        trace_zs.append((np.array(trace[2]) * v2ef).astype(np.float32).tolist())

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

    tefield.du_id = du_id
    tefield.du_seconds = du_seconds
    tefield.du_nanoseconds = du_nanoseconds
    tefield.trigger_position = trigger_position
    tefield.trigger_flag = trigger_flag
    tefield.atm_temperature = atm_temperature
    tefield.atm_pressure = atm_pressure
    tefield.atm_humidity = atm_humidity
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

# write the tree to the storage, but don't close the file - it will be used for tshower
# ToDo: is this correct? Not sure if I should use the file opened for writing when I am reading
tefield.write(filename, close_file=False)
print("Wrote tefield")

# Generation of shower data for each event - this should be reonstruction, but here just dumb values
tshower = ShowerEventTree()
tshower.comment = "Generated DataStoringExample.py"
# Loop through all Efield entries
for i in range(tefield.get_entries()):
    # Get the Efield event
    tefield.get_entry(i)

    tshower.run_number = tefield.run_number
    tshower.event_number = tefield.event_number

    tshower.shower_type = "particle"
    tshower.shower_energy = np.random.random(1) * 1e8
    tshower.shower_azimuth = np.random.random(1) * 360
    tshower.shower_zenith = np.random.random(1) * 180 - 90
    tshower.shower_core_pos = np.random.random(3)
    tshower.atmos_model = "dense air dummy"
    tshower.atmos_model_param = np.random.random(3)
    tshower.magnetic_field = np.random.random(3)
    tshower.date = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    tshower.ground_alt = 3000.0 + np.random.randint(0, 1000)
    tshower.xmax_grams = np.random.random(1) * 500
    tshower.xmax_pos_shc = np.random.random(3)
    tshower.xmax_alt = np.random.randint(3000, 5000) * 1.0
    tshower.gh_fit_param = np.random.random(3)
    tshower.core_time = np.random.randint(0, 10000) * 1.0

    tshower.fill()

tshower.write(filename)
print("Wrote tshower")

print(f"Finished writing file {filename}")
