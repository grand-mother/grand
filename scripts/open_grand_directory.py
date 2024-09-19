#! /usr/bin/env python3
# Opens the GRAND ROOT directory with a DataDirectory class and leaves the prompt open, so the user can work with the opened directory

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

# Construct the command based on the arguments
command = f"from grand.dataio.root_trees import *; d = DataDirectory('{args.dirname}');"
if not args.s:
    command+=f" print(f'\\n\\033[0;31mOpened directory {args.dirname} as d\\033[0m\\n'); d.print(verbose={verbose})"
 
os.execlp(interp, interp, '-i', '-c', command)

