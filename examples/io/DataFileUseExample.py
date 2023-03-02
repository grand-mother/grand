#!/usr/bin/python
# An example of using DataFile for ROOT file reading
import numpy as np
import sys
from grand.io.root_trees import *

# Need to provide a file to read
if len(sys.argv)<2:
    print("Please provide a ROOT file name to read")
    exit()

df = DataFile(sys.argv[1])

df.print()

# Get the first event in the TVoltage tree
df.tvoltage.get_entry(0)
# Print the run_number and evt_number for the first event
print(f"\nRead run number {df.tvoltage.run_number}, event number {df.tvoltage.event_number} of tvoltage")
# Print the trace X for du 0 for this event
print("Trace X for du 0 for tvoltage:")
print(df.tvoltage.trace_x[0])
