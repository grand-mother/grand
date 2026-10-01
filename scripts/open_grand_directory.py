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
# The name reaches the shell through the environment, never as code (#184)
os.environ["GRAND_OPEN"] = args.dirname
command = "import os; from grand.dataio import *; d = DataDirectory(os.environ['GRAND_OPEN']);"
if not args.s:
    command+=" print('\\n\\033[0;31mOpened directory %%s as d\\033[0m\\n' %% os.environ['GRAND_OPEN']); d.print(verbose=%s)" % verbose
 
os.execlp(interp, interp, '-i', '-c', command)

