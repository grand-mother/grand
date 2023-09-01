#!/usr/bin/python
# Example of Event generation and storing with GRANDlib classes
# For now it creates an event with _meaningless dummy_ data
import sys
from grand.grandlib_classes.grandlib_classes import *

# The file to store to (preferably should end with ".root" extension)
if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = "dummy_example_events.root"

# How many events to generate?
event_count = 10

# Loop through events
for event_number in range(event_count):
    print("Event ", event_number)
    # auto_file_close=False prevents opening/writing to HDD/closing separately for each event - a speed up, but a close call has to be called manually after the loop
    event = Event(auto_file_close=False)

    # Fill in the Run part

    event.run_number = 0
    event.event_number = event_number
    event.site = "dummy site"
    event.first_event = 0
    event.last_event = event_count
    # ToDo: this should be a proper list
    event._t_bin_size = [2]

    # Fill the voltage and antenna part with _dummy random_ data
    event.voltages = []
    event.antennas = []
    event.efields = []
    # Number of traces in this event
    trace_count = np.random.randint(3, 7)
    for i in range(trace_count):
        # The voltage part
        v = Voltage()
        # The trace length
        v.n_points = np.random.randint(900, 1000)
        # v.n_points = np.random.randint(9, 10)
        # v.trace_x = np.random.randint(-200, 201, v.n_points)/100.
        # v.trace_y = np.random.randint(-200, 201, v.n_points)/100.
        # v.trace_z = np.random.randint(-200, 201, v.n_points)/100.
        v.trace = [np.random.randint(-200, 201, v.n_points) / 100., np.random.randint(-200, 201, v.n_points) / 100., np.random.randint(-200, 201, v.n_points) / 100.]
        v.du_id = i
        event.voltages.append(v)

        # The antenna part
        a = Antenna()
        a.atm_temperature = np.random.randint(-400, 401)/100.
        a.atm_pressure = np.random.randint(9000, 11000)/10.
        a.atm_humidity = np.random.rand()*100
        a.battery_level = np.random.rand()*100
        a.firmware_version = 1
        a.id = i
        event.antennas.append(a)

        # The efield part
        e = Efield()
        e.n_points = v.n_points
        v2ef = 1.17
        e.trace = [v.trace.x*v2ef, v.trace.y*v2ef, v.trace.z*v2ef]
        # e.trace_x = v.trace_x*v2ef
        # e.trace_y = v.trace_y*v2ef
        # e.trace_z = v.trace_z*v2ef
        e.du_id = i
        event.efields.append(e)

    # The shower part
    event.shower = Shower()
    event.shower.energy = np.random.rand()
    ## Shower Xmax [g/cm2]
    event.shower.Xmax = np.random.randint(1000, 4000)/10.
    ## Shower position in the site's reference frame
    event.shower.Xmaxpos = np.random.rand(3)*1000
    ## Direction of origin (ToDo: is it the same as origin of the coordinate system?)
    event.shower.origin_geoid = np.zeros(3)
    ## Poistion of the core on the ground in the site's reference frame
    event.shower.core_ground_pos = np.random.rand(4)*1000

    event.write(filename)
    print(f"Wrote event {event_number}")

# Close all the files opened by the event with auto_file_close=False
event.close_files()
