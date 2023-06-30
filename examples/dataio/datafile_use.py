#!/usr/bin/python
# An example of using DataFile for ROOT file reading
import numpy as np
import sys
from grand.dataio.root_trees import *

# Need to provide a file to read
if len(sys.argv)<2:
    print("Please provide a ROOT file name to read")
    exit()

df = DataFile(sys.argv[1])

df.print()

# Get the first event in the TVoltage tree
# Try for tvoltage. If it does not exist, try for trawvoltage
if hasattr(df, "tvoltage"):
    df.tvoltage.get_entry(0)

    # Print the run_number and evt_number for the first event
    print(f"\nRead run number {df.tvoltage.run_number}, event number {df.tvoltage.event_number} of tvoltage")

    # Print the trace X for du 0 for this event
    print("Trace X for du 0 for tvoltage:")
    print(df.tvoltage.trace[0][0])

elif hasattr(df, "trawvoltage"):
    df.trawvoltage.get_entry(0)

    # Print the run_number and evt_number for the first event
    print(f"\nRead run number {df.trawvoltage.run_number}, event number {df.trawvoltage.event_number} of trawvoltage")

    print("Trace X for du 0 for trawvoltage:")
    print(df.trawvoltage.trace_ch[0][0])

elif hasattr(df, "tefield"):
    df.tefield.get_entry(0)

    # Print the run_number and evt_number for the first event
    print(f"\nRead run number {df.tefield.run_number}, event number {df.tefield.event_number} of tefield")

    # Print the trace X for du 0 for this event
    print("Trace X for du 0 for tefield:")
    print(df.tefield.trace[0][0])

