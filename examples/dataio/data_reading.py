#!/usr/bin/python
'''
An example of reading data from a file
'''
import numpy as np
import sys
from grand.dataio.root_trees import *

if len(sys.argv) < 2:
    tadccounts = TADC("stored_data.root")
else:
    print(f"Reading file {sys.argv[1]}")
    tadccounts = TADC(sys.argv[1])

# Get the list of runs,events
list_of_events = tadccounts.get_list_of_events()
# If there are no events, exit
if len(list_of_events) == 0:
    raise Exception("No events in the tadcounts tree.")

# Print out the list of runs,events
tadccounts.print_list_of_events()
# Read the first event from the list of events
tadccounts.get_event(*list_of_events[0])
print(
    "ADCCounts first event readout: tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_ch0[0]"
)
print(tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_ch[0])
print(tadccounts.trace_ch[0][0])

print("Iterate through ADCCounts")
print("Entry #, Event #, Run #")
for i, en in enumerate(tadccounts):
    print(i, en.event_number, en.run_number)

if len(sys.argv) < 2:
    tefield = TEfield("stored_data.root")
else:
    tefield = TEfield(sys.argv[1])

list_of_events = tefield.get_list_of_events()
tefield.get_event(*list_of_events[-1])
print(
    "\nEfield readout: tefield.event_number, tefield.det_time[0], tefield.trace[0][0], tadccounts.evt_id"
)
print(
    "The event_number of tadccounts changed to 4 when tefield event with event_number 4 was requested"
)
print(tefield.event_number, tefield.du_seconds[0], tefield.trace[0][0], tadccounts.event_number)

# Hard exit to avoid ROOT crash on defriending - not needed for ROOT 6.26.02 and above
if ROOT.gROOT.GetVersionInt() < 62602:
    import os

    os._exit(1)
