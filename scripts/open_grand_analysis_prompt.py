#! /usr/bin/env python3
# Opens the GRAND ROOT directory with an EventList class and leaves the prompt open, so the user can work with the events

import argparse
import os
import sys

# Create the argument parser
parser = argparse.ArgumentParser(description='Open a GRAND directory in an IPython or Python shell.')

# Add the command-line options
parser.add_argument('-p', action='store_true', help='Use Python instead of IPython')
parser.add_argument('-s', action='store_true', help='Do not print any initial output')
parser.add_argument('-nv', action='store_true', help='Do not print verbose output')
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

# Read the file name from command line
if len(sys.argv) > 1:
    dir_name = sys.argv[1]
else:
    print("Please provide a GRAND data output directory")
    exit()

print("Reading directory", dir_name)

# Construct the command based on the arguments
command = f"from grand.grandlib_classes.grandlib_classes import *; el = EventList('{args.dirname}');"
if not args.s:
    command+=f" print(f'\\n\\033[0;31mOpened directory {args.dirname} as d\\033[0m\\n');"
    command += " print('You can now iterate through events with, for example:\\n\\nfor i,e in enumerate(el):\\n  print(e.event_number)\\n  ...')"
 
os.execlp(interp, interp, '-i', '-c', command)

