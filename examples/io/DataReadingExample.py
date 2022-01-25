#!/usr/bin/python
# An example of reading data from a file
import numpy as np
from grand.io.root_trees import *
# import ROOT

tadccounts = ADCCountsTree("stored_data.root")

tadccounts.get_event(2)
print("ADCCounts readout: tadccounts.evt_id, tadccounts.det_time[0], tadccounts.trace_x[0]")
print(tadccounts.evt_id, tadccounts.det_time[0], tadccounts.trace_x[0])


tefield = EfieldTree("stored_data.root")
#tvoltage = GRANDVoltageTree("stored_data.root")

tefield.get_event(2)
print("\nEfield readout: tefield.evt_id, tefield.det_time[0], tefield.trace_x[0][0], tadccounts.evt_id")
print("The evt_id of tadccounts changed to 4 when tefield event with evt_id 4 was requested")
print(tefield.evt_id, tefield.det_time[0], tefield.trace_x[0][0], tadccounts.evt_id)

