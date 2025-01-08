#! /usr/bin/env python3
# Opens the GRAND ROOT file with a DataFile class and leaves the prompt open, so the user can work with the opened file

import argparse
import os
import sys

# Create the argument parser
parser = argparse.ArgumentParser(description='Open a GRAND file in an IPython or Python shell.')

# Add the command-line options
parser.add_argument('-p', action='store_true', help='Use Python instead of IPython')
parser.add_argument('-s', action='store_true', help='Do not print any initial output')
parser.add_argument('filename', metavar='<filename>', type=str, help='The GRAND ROOT filename to load')

# Parse the arguments
args = parser.parse_args()

interp = "ipython"

# Prepare to run in the standard Python shell if requested
if args.p:
    interp = "python"

# Construct the command based on the arguments
command = f"from grand.dataio.root_trees import *; f = DataFile('{args.filename}');"
if not args.s:
    command+=f" print(f'\\n\\033[0;31mOpened file {args.filename} as f\\033[0m\\n'); f.print()"
os.execlp(interp, interp, '-i', '-c', command)

