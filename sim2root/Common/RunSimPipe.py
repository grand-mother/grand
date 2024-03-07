#!/usr/bin/env $ZHAIRESPYTHON
import sys
import os
import logging   #for...you guessed it...logging
logging.basicConfig(level=logging.DEBUG)
import argparse  #for command line parsing
import glob      #for listing files in directories
import subprocess#for launching the script or the qsub

try:
  PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]
except:
  logging.debug("PYTHONINTERPRETER not defined, defaulting to python")
  PYTHONINTERPRETER="python"

#Manual Configuration

PRODUCEGRANDROOT="./sim2root.py"
PRODUCEVOLTAGE="../../scripts/convert_efield2voltage.py"
PRODUCEADC="../../scripts/convert_voltage2adc.py"
PRODUCEDC2Efield="../../scripts/convert_efield2efield.py"

INPUTDIR="../ZHAireSRawRoot"
EXTRA="ZHAireS"

#No command line parameters for now.
""" 
parser = argparse.ArgumentParser(description='A script to run the simulation pipe on a directory containing rawroot files')
parser.add_argument('InputDir', #name of the parameter
                    metavar="InputDir", #name of the parameter value in the help
                    help='Input Directory, where the rawroot files are') # help message for this parameter
parser.add_argument('Extra', #name of the parameter
                    metavar="Extra", #name of the parameter value in the help
                    help='Extra info you want to append at the end of the directory name in the output') # help message for this parameter
"""

#########################################################################################################################################################
# Grandroot
########################################################################################################################################################
logging.debug(" Trying to make GrandRoot file")
#line to make file
# python  /home/mjtueros/TrabajoTemporario/docker/grand/sim2root/Common/sim2root.py /home/mjtueros/TrabajoTemporario/docker/simpipe/DC2Outbox/rawrootfiles -e GP300
cmd=PYTHONINTERPRETER+" "+PRODUCEGRANDROOT+" "+INPUTDIR+" -e "+EXTRA
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
#print(stdout)
print(stderr)


#########################################################################################################################################################
# Voltage
########################################################################################################################################################
logging.debug(" Trying to produce voltages")
#since we dont know where the output will be created (becouse its automatically done by the sim2root, we will take the latest directory produced
INPUTDIR=max(glob.glob('*/'), key=os.path.getmtime)
OUTPUTFILE=glob.glob(INPUTDIR+"/*efield_*L0*.root")
OUTPUTFILE=OUTPUTFILE[0].replace("efield", "voltage")
OUTPUTFILE=OUTPUTFILE[:-5]

#no noise and no chain
#cmd=PYTHONINTERPRETER+" "+PRODUCEVOLTAGE+" "+INPUTDIR+" --seed 1234  --target_duration_us=4.096 --add_jitter_ns  5 --calibration_smearing_sigma 0.075 --verbose=info -o " + OUTPUTFILE+"-no_rf_chain-no_noise.root --no_rf_chain --no_noise"
#no noise
#cmd=PYTHONINTERPRETER+" "+PRODUCEVOLTAGE+" "+INPUTDIR+" --seed 1234  --target_duration_us=4.096 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --verbose=info -o " + OUTPUTFILE+"-no_noise.root --no_noise"
#no chain
#cmd=PYTHONINTERPRETER+" "+PRODUCEVOLTAGE+" "+INPUTDIR+" --seed 1234  --target_duration_us=4.096 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --verbose=info -o " + OUTPUTFILE+"-no_rf_chain.root --no_rf_chain" 
#the real thing
cmd=PYTHONINTERPRETER+" "+PRODUCEVOLTAGE+" "+INPUTDIR+" --seed 1234 --target_duration_us=4.096 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --verbose=info -o " + OUTPUTFILE+".root"
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
#print(stdout)
print(stderr)


#########################################################################################################################################################
# ADC
#####################################################################################################################################################
logging.debug(" Trying to produce ADCs")
cmd=PYTHONINTERPRETER+" "+PRODUCEADC+" "+INPUTDIR
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes.
print(stdout)
print(stderr)


#########################################################################################################################################################
# DC2Efields
#####################################################################################################################################################
logging.debug(" Trying to produce DC2efields") 
# python /home/mjtueros/TrabajoTemporario/docker/grand/scripts/convert_efield2efield.py ./sim_Xiaodushan_20221026_000000_RUN0_CD_GP300_0000 --add_noise_uVm 22 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --target_duration_us 4.096 --target_sampling_rate_mhz 500
cmd=PYTHONINTERPRETER+" "+PRODUCEDC2Efield+" "+INPUTDIR+"  --add_noise_uVm 22 --add_jitter_ns 5 --calibration_smearing_sigma 0.075 --target_duration_us 4.096 --target_sampling_rate_mhz 500"
print("about to run:" + cmd)
p = subprocess.Popen(cmd,cwd=".",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
stdout,stderr=p.communicate() #the communicate will make it to wait until it finishes

print(stdout)
print(stderr)
