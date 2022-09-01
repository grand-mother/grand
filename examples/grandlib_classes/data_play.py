#!/usr/bin/python
# Example use of the grandlib classes for stored data read-out
import sys
from grand.grandlib_classes.grandlib_classes import *

e = Event()

# Readout the event from file
if len(sys.argv)==2:
    e.file = sys.argv[1]
    e.run_number = 0
    e.event_number = 0
    e.fill_event_from_trees()
    e.print()
    print(e.data_source)
    for i,v in enumerate(e.voltages):
        print(i, v.trace_x)
    for i,ef in enumerate(e.efields):
        print(i, ef.trace_x)

    # Hard exit to avoid ROOT crash on defriending - not needed for ROOT 6.26.02 and above
    if ROOT.gROOT.GetVersionInt() < 62602:
        import os
        os._exit(1)
    e.file.Close()

else:
    print("Please provide a ROOT filename with the trees")
