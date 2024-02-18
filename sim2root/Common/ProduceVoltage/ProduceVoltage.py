#!/usr/env python 
#$ -j y
#$ -notify
#$ -l ct=0:30:00
#$ -l vmem=0.5G
#$ -l fsize=0.5G
#$ -l sps=1
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import subprocess
import datetime
import time
import sys
import logging
import os
import glob
#ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
#sys.path.append(ZHAIRESPYTHON)
#PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]


'''
generate raw root from zhaires sym
'''

#Python=PYTHONINTERPRETER
#efield2volt="/pbs/home/t/tueros/GRANDlib/grand/scripts/convert_efield2voltage.py"
Python="python"
#/home/mjtueros/TrabajoTemporario/docker/grand/scripts/convert_efield2voltage.py
#/home/mjtueros/TrabajoTemporario/docker/grand/sim2root/Common/ProduceVoltage/ProduceVoltage.py
efield2volt =  os.path.abspath(os.path.dirname(__file__)) + "/../../../scripts/convert_efield2voltage.py"

def VoltGRANDRoot(InputFile, OutputFile):

  #RENAME THE FILE (not working)
  #head,tail=os.path.split(InputFile)
  #renamed=head+".root"

  #cmd="mv " +  InputFile + " " + renamed
  #print("About to run:"+ cmd)
  #p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  #stdout,stderr=p.communicate()
  
  #Use event number as seed
  Seed= str(int(time.clock_gettime(1)))

  #regular
  OutputFile1=OutputFile+"_with-rf_with-noise.root"
  cmd=Python + " " + efield2volt + " " + InputFile + " --seed " + Seed +" --verbose error --target_sampling_rate_mhz=500 --target_duration_us=4.096" + " -o " + OutputFile1
  print("About to run:"+ cmd)
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  stdout,stderr=p.communicate()
  print("errors:")
  print(stderr)
  print("output:")
  print(stdout)

  #no-noise
  OutputFile1=OutputFile+"_with-rf_no-noise.root"
  cmd=Python + " " + efield2volt + " " + InputFile + " --seed " + Seed +" --no_noise --verbose error --target_sampling_rate_mhz=500 --target_duration_us=4.096" + " -o " + OutputFile1
  print("About to run:"+ cmd)
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  stdout,stderr=p.communicate()
  print("errors:")
  print(stderr)
  print("output:")
  print(stdout)  
  

  #no-rf
  OutputFile1=OutputFile+"_no-rf_with-noise.root"
  cmd=Python + " " + efield2volt + " " + InputFile + " --seed " + Seed +" --no_rf_chain --verbose error --target_sampling_rate_mhz=500 --target_duration_us=4.096" + " -o " + OutputFile1
  print("About to run:"+ cmd)
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  stdout,stderr=p.communicate()
  print("errors:")
  print(stderr)
  print("output:")
  print(stdout)  


  
  #no-rf  -no-noise
  OutputFile1=OutputFile+"_no-rf_no-noise.root"
  cmd=Python + " " + efield2volt + " " + InputFile + " --seed " + Seed +" --no_noise --no_rf_chain --verbose error --target_sampling_rate_mhz=500 --target_duration_us=4.096" + " -o " + OutputFile1
  print("About to run:"+ cmd)
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  stdout,stderr=p.communicate()
  print("errors:")
  print(stderr)
  print("output:")
  print(stdout)  




  ##################################3


def main():

    logging.basicConfig(level=logging.DEBUG)
    if ( len(sys.argv)<3 ):
        print("""

            Usage: python3  <inputdirectory> <outputdirectory>

        """)
        sys.exit(0)

    inputdirectory = sys.argv[1]
    outputdirectory = sys.argv[2]

    efieldfile=glob.glob(inputdirectory+"/efield_*.root")[0]

    head,tail=os.path.split(efieldfile)

    InputJobName=os.path.splitext(tail)[0]

    OutFileName=outputdirectory+"/"+"voltage_"+InputJobName[8:]

    logging.info("About to produce voltage for GRANDRoot file from "+inputdirectory)
    logging.debug("output file will be in:"+OutFileName)

    VoltGRANDRoot(inputdirectory, OutFileName)





print(__name__)
if __name__ == '__main__':
 main()

