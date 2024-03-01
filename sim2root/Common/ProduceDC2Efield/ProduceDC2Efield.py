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
generate DC2efield from efield
'''

#Python=PYTHONINTERPRETER
Python="python"
efield2efield =  os.path.abspath(os.path.dirname(__file__)) + "/../../../scripts/convert_efield2efield.py"

def VoltGRANDRoot(InputDirectory, OutputDirectory):

  #cmd="mv " +  InputFile + " " + renamed
  #print("About to run:"+ cmd)
  #p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  #stdout,stderr=p.communicate()
  
  #Use event number as seed
  Seed= str(int(time.clock_gettime(1)))

  cmd=Python + " " + efield2efield + " " + InputDirectory + " --seed " + Seed +" --verbose error --add_jitter_ns 5 --add_noise_uVm 22  --target_sampling_rate_mhz=500 --target_duration_us=4.096" + " -od " + outputdiectory
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

    logging.info("About to produce voltage for GRANDRoot file from "+inputdirectory)
    logging.debug("output file will be in:"+ outputdirectory)

    DC2Efield(inputdirectory, outputdirectory)





print(__name__)
if __name__ == '__main__':
 main()

