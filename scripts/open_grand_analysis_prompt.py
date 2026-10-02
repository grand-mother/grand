#! /usr/bin/env python3
# Opens the GRAND ROOT directory with an EventList class and leaves the prompt open, so the user can work with the events

import argparse
import os
import sys

# Create the argument parser
parser = argparse.ArgumentParser(description='Open a GRAND directory as an EventList in an IPython or Python shell.')

# Add the command-line options
parser.add_argument('-p', action='store_true', help='Use Python instead of IPython')
parser.add_argument('-s', action='store_true', help='Do not print any initial output')
parser.add_argument('-nv', action='store_true', help='Do not print verbose output')
parser.add_argument('-trv', "--use_trawvoltage", action='store_true', help='Use TRawVoltage instead of TVoltage')
parser.add_argument('dirname', metavar='<dirname>', type=str, help='The GRAND ROOT directory to load')

# Parse the arguments
args = parser.parse_args()

interp = "ipython"

# Prepare to run in the standard Python shell if requested
if args.p:
    interp = "python"

if args.nv:
    verbose=False
else:
    verbose=True

# argparse requires the directory; sys.argv[1] could be an option such as -p
print("Reading directory", args.dirname)

# Construct the command based on the arguments
# The name reaches the shell through the environment, never as code (#184)
os.environ["GRAND_OPEN"] = args.dirname
command = "import os; from grand.aoi import *; el = EventList(os.environ['GRAND_OPEN'], use_trawvoltage=%s);" % bool(args.use_trawvoltage)
if not args.s:
    command+=" print('\\n\\033[0;31mCreated a list of events in directory %s as el\\033[0m\\n' % os.environ['GRAND_OPEN']);"
    command += " print('You can now iterate through events with, for example:\\n\\nfor i,e in enumerate(el):\\n  print(e.event_number)\\n  ...')"
 
os.execlp(interp, interp, '-i', '-c', command)

