#!/usr/bin/env $ZHAIRESPYTHON
import sys
import os
import logging   #for...you guessed it...logging
logging.basicConfig(level=logging.DEBUG)
import argparse  #for command line parsing
import glob      #for listing files in directories
from pipeline_step import run_step  # runs each step with its output shown live (#121)
import shlex

try:
  PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]
except:
  logging.debug("PYTHONINTERPRETER not defined, defaulting to python")
  PYTHONINTERPRETER="python"
PY=shlex.split(PYTHONINTERPRETER)  # each step runs from an argument list, without a shell

#Manual Configuration

PRODUCEGRANDROOT="./sim2root.py"
PRODUCEVOLTAGE="../../scripts/convert_efield2voltage.py"
PRODUCEADC="../../scripts/convert_voltage2adc.py"
PRODUCEDC2Efield="../../scripts/convert_efield2efield.py"

parser = argparse.ArgumentParser(description='A script to run the simulation pipe on a directory containing rawroot files')
parser.add_argument('InputDir', #name of the parameter
                    metavar="InputDir", #name of the parameter value in the help
                    default=None,
                    help='Input Directory, where the rawroot files are',) # help message for this parameter
parser.add_argument('Extra', #name of the parameter
                    metavar="Extra", #name of the parameter value in the help
                    default=None,
                    help='Extra info you want to append at the end of the directory name in the output',) # help message for this parameter

# sim2root requires the site layout; without it the first step failed and the
# next steps ran on whichever directory was newest (#221)
parser.add_argument("-sl", "--site_layout", required=True,
                    help="The layout of the site, passed on to sim2root (eg. GP13, GP80, GP300, GAA)")

args=parser.parse_args()

if args.InputDir is not None:
        INPUTDIR=args.InputDir

if args.Extra is not None:
        EXTRA=args.Extra

#########################################################################################################################################################
# Grandroot
########################################################################################################################################################
logging.debug(" Trying to make GrandRoot file")
#line to make file
# Directories present before the step: its output is the one that is new
BEFORE=set(glob.glob('*/'))
cmd=PY+[PRODUCEGRANDROOT, INPUTDIR, "--target_duration_us=4.096", "--trigger_time_ns", "800", "-sl", str(args.site_layout), "-e", EXTRA]
if run_step(cmd) != 0:
    sys.exit("sim2root failed; stopping before the next steps (#221)")



#########################################################################################################################################################
# Voltage
########################################################################################################################################################
logging.debug(" Trying to produce voltages")
#since we dont know where the output will be created (becouse its automatically done by the sim2root, we will take the latest directory produced
NEW=set(glob.glob('*/'))-BEFORE
if not NEW:
    sys.exit("sim2root produced no new directory; stopping rather than processing an older one (#221)")
INPUTDIR=max(NEW, key=os.path.getmtime)
OUTPUTFILE=glob.glob(INPUTDIR+"/*efield_*L0*.root")
# A bare name: the script writes it into INPUTDIR already, and a path here
# was doubled (INPUTDIR/INPUTDIR/...), so the voltage step failed (#221)
OUTPUTFILE=os.path.basename(OUTPUTFILE[0]).replace("efield", "voltage")
OUTPUTFILE=OUTPUTFILE[:-5]

#the "real" thing
cmd=PY+[PRODUCEVOLTAGE, INPUTDIR, "--seed", "1234", "--verbose=info", "--add_jitter_ns", "5", "--calibration_smearing_sigma", "0.075", "-o", OUTPUTFILE+".root"]
if run_step(cmd) != 0:
    sys.exit("a step failed; stopping before the next ones (#221)")


#########################################################################################################################################################
# ADC
#####################################################################################################################################################
logging.debug(" Trying to produce ADCs")
cmd=PY+[PRODUCEADC, INPUTDIR]
if run_step(cmd) != 0:
    sys.exit("a step failed; stopping before the next ones (#221)")


#########################################################################################################################################################
# DC2Efields
#####################################################################################################################################################
logging.debug(" Trying to produce DC2efields") 
cmd=PY+[PRODUCEDC2Efield, INPUTDIR, "--add_noise_uVm", "22", "--add_jitter_ns", "5", "--calibration_smearing_sigma", "0.075", "--target_duration_us", "4.096", "--target_sampling_rate_mhz", "500"]
if run_step(cmd) != 0:
    sys.exit("a step failed; stopping before the next ones (#221)")

