#!/usr/bin/env $ZHAIRESPYTHON
import sys
import os
import logging   #for...you guessed it...logging
logging.basicConfig(level=logging.DEBUG)
import argparse  #for command line parsing
import glob      #for listing files in directories
from pipeline_step import run_step  # runs each step with its output shown live (#121)
import shlex
from pathlib import Path

try:
  PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]
except:
  logging.debug(" PYTHONINTERPRETER not defined, defaulting to python")
  PYTHONINTERPRETER="python"
PY=shlex.split(PYTHONINTERPRETER)  # each step runs from an argument list, without a shell

#Manual Configuration

current_path = Path(__file__).parent

PRODUCEGRANDROOT = str(current_path / "sim2root.py")
PRODUCEVOLTAGE = str(current_path / "../../scripts/convert_efield2voltage.py")
PRODUCEADC = str(current_path / "../../scripts/convert_voltage2adc.py")
PRODUCEDC2Efield = str(current_path / "../../scripts/convert_efield2efield.py")



parser = argparse.ArgumentParser(description='A script to run the simulation pipe on a directory containing rawroot files')
parser.add_argument('InputDir', #name of the parameter
                    metavar="InputDir", #name of the parameter value in the help
                    default=None,
                    help='Input Directory, where the rawroot files are',) # help message for this parameter
parser.add_argument('Extra', #name of the parameter
                    metavar="Extra", #name of the parameter value in the help
                    default=None,
                    help='Extra info you want to append at the end of the directory name in the output',) # help message for this parameter
parser.add_argument("-sl", "--site_layout", help="The layout of the site (eg. GP13, GP80, GAA)", default=None, required=True)

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
cmd=PY+[PRODUCEGRANDROOT, INPUTDIR, "--target_duration_us=2.048", "--trigger_time_ns", "550", "-sl", str(args.site_layout), "-e", EXTRA]
run_step(cmd)


#########################################################################################################################################################
# Voltage
########################################################################################################################################################
logging.debug(" Trying to produce voltages")
#since we dont know where the output will be created (becouse its automatically done by the sim2root, we will take the latest directory produced
INPUTDIR=max(glob.glob('*/'), key=os.path.getmtime)
OUTPUTFILE=glob.glob(INPUTDIR+"/*efield_*L0*.root")
OUTPUTFILE=OUTPUTFILE[0].replace("efield", "voltage")
OUTPUTFILE=OUTPUTFILE[:-5]
OUTPUTFILE = str(Path(OUTPUTFILE).name)

#we dont add galactic noise, becouse ADC noise already has that!
cmd=PY+[PRODUCEVOLTAGE, INPUTDIR, "--seed", "1234", "--verbose=info", "--add_jitter_ns", "5", "--calibration_smearing_sigma", "0.075", "--no_noise", "-o", OUTPUTFILE+".root"]
run_step(cmd)


#########################################################################################################################################################
# ADC
#####################################################################################################################################################
logging.debug(" Trying to produce ADCs")
cmd=PY+[PRODUCEADC, INPUTDIR, "--add_noise_from", f"{current_path}/LongNoiseTraces/", "--seed", "1234"]
run_step(cmd)


#########################################################################################################################################################
# DC2Efields
#####################################################################################################################################################
logging.debug(" Trying to produce DC2efields") 

cmd=PY+[PRODUCEDC2Efield, INPUTDIR, "--add_noise_uVm", "22", "--add_jitter_ns", "5", "--seed", "1234", "--calibration_smearing_sigma", "0.075", "--target_duration_us", "2.048", "--target_sampling_rate_mhz", "500"]
run_step(cmd)

