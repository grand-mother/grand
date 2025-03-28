#!/usr/bin/env $ZHAIRESPYTHON
import sys
import os
import logging   #for...you guessed it...logging
logging.basicConfig(level=logging.DEBUG)
import argparse  #for command line parsing
import glob      #for listing files in directories
import subprocess#for launching the script or the qsub
from pathlib import Path

try:
  PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]
except:
  logging.debug(" PYTHONINTERPRETER not defined, defaulting to python")
  PYTHONINTERPRETER="python"

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
cmd=PYTHONINTERPRETER+" "+PRODUCEGRANDROOT+" "+INPUTDIR+f" --target_duration_us=2.048 --trigger_time_ns 550 -sl {args.site_layout} -e "+EXTRA
print("about to run:" + cmd)
# p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()  # Ensure immediate output
p.wait()
# stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
#print(stdout)
# print(stderr)

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
cmd=PYTHONINTERPRETER+" "+PRODUCEVOLTAGE+" "+INPUTDIR+" --seed 1234 --verbose=info --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --no_noise -o " + OUTPUTFILE+".root"
print("about to run:" + cmd)
# p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()  # Ensure immediate output
p.wait()
# stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
#print(stdout)
# print(stderr)


#########################################################################################################################################################
# ADC
#####################################################################################################################################################
logging.debug(" Trying to produce ADCs")
cmd=PYTHONINTERPRETER+" "+PRODUCEADC+" "+INPUTDIR +f" --add_noise_from {current_path}/LongNoiseTraces/  --seed 1234"
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()  # Ensure immediate output
p.wait()

# stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
# print(stdout)
# print(stderr)

#########################################################################################################################################################
# DC2Efields
#####################################################################################################################################################
logging.debug(" Trying to produce DC2efields") 

cmd=PYTHONINTERPRETER+" "+PRODUCEDC2Efield+" "+INPUTDIR+" --add_noise_uVm 22 --add_jitter_ns 5 --seed 1234 --calibration_smearing_sigma 0.075 --target_duration_us 2.058 --target_sampling_rate_mhz 500"
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()  # Ensure immediate output
p.wait()

# stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes

# print(stdout)
# print(stderr)
