#!/usr/bin/python
# Example use of the grandlib classes for stored data read-out
import sys
from grand.grandlib_classes.grandlib_classes import *

# Avoid printing out whole traces
np.set_printoptions(threshold=5)

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
        print(i, np.array(v.trace)[:,0])
    for i,ef in enumerate(e.efields):
        print(i, np.array(ef.trace)[:,0])

    # Hard exit to avoid ROOT crash on defriending - not needed for ROOT 6.26.02 and above
    if ROOT.gROOT.GetVersionInt() < 62602:
        import os
        os._exit(1)
    e.file.Close()

else:
    print("Please provide a ROOT filename with the trees")

