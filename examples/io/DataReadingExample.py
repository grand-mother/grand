#!/usr/bin/python
# An example of reading data from a file
import numpy as np
import sys
from grand.io.root_trees import *

if len(sys.argv) < 2:
    tadccounts = ADCEventTree("stored_data.root")
else:
    print(f"Reading file {sys.argv[1]}")
    tadccounts = ADCEventTree(sys.argv[1])

# Print out the list of runs,events
tadccounts.print_list_of_events()
# Get the list of runs,events
list_of_events = tadccounts.get_list_of_events()
# Read the first event from the list of events
tadccounts.get_event(*list_of_events[0])
print("ADCCounts readout: tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_0[0]")
print(tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_0[0])
print(tadccounts.trace_0[0][0])

if len(sys.argv) < 2:
    tefield = EfieldEventTree("stored_data.root")
else:
    tefield = EfieldEventTree(sys.argv[1])
# tvoltage = GRANDVoltageTree("stored_data.root")

list_of_events = tefield.get_list_of_events()
tefield.get_event(*list_of_events[-1])
print(
    "\nEfield readout: tefield.event_number, tefield.det_time[0], tefield.trace_x[0][0], tadccounts.evt_id"
)
print(
    "The event_number of tadccounts changed to 4 when tefield event with event_number 4 was requested"
)
print(tefield.event_number, tefield.du_seconds[0], tefield.trace_x[0][0], tadccounts.event_number)

# Hard exit to avoid ROOT crash on defriending - not needed for ROOT 6.26.02 and above
if ROOT.gROOT.GetVersionInt() < 62602:
    import os

    os._exit(1)
