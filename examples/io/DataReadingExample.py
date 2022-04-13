#!/usr/bin/python
# An example of reading data from a file
import numpy as np
import sys
from grand.io.root_trees import *

if len(sys.argv)<2:
    tadccounts = ADCEventTree("stored_data.root")
else:
    tadccounts = ADCEventTree(sys.argv[1])

#tadccounts.get_event(10002, 2201)
tadccounts.get_event(2)
print("ADCCounts readout: tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_0[0]")
print(tadccounts.event_number, tadccounts.time_seconds, tadccounts.trace_0[0])


tefield = EfieldEventTree("stored_data.root")
#tvoltage = GRANDVoltageTree("stored_data.root")

# tefield.get_event(10002, 2201)
tefield.get_event(2)
print("\nEfield readout: tefield.event_number, tefield.det_time[0], tefield.trace_x[0][0], tadccounts.evt_id")
print("The event_number of tadccounts changed to 4 when tefield event with event_number 4 was requested")
print(tefield.event_number, tefield.du_seconds[0], tefield.trace_x[0][0], tadccounts.event_number)

# Hard exit to avoid ROOT crash on defriending - not needed for ROOT 6.26.02 and above
if ROOT.gROOT.GetVersionInt() < 62602:
    import os
    os._exit(1)

